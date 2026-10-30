#!/usr/bin/env python3
"""Independent verifier that a constructed number is an Euler pseudoprime to many bases.

Self-contained on purpose: it imports nothing from ``pdc``, so it does not trust the
construction in :mod:`pdc.euler` in any way.  Input is a factored file (one prime per line,
``#`` comments and blank lines ignored) giving the prime factors of a single number ``n``,
plus the modulus ``Lambda``.

What it checks
--------------
1. Every factor is prime (strong probable-prime by default; ``--full-primality`` gives a
   rigorous Lucas/Pocklington proof, valid because ``p-1 | Lambda`` is fully factored).
2. ``n`` is a Carmichael number: squarefree, ``>= 3`` factors, ``(p-1) | Lambda`` for every
   ``p``, and ``n == 1 (mod Lambda)`` (residue product) -- Korselt via Lambda, no giant ``n``.
3. Its class: A (``v_2(n-1) > max_p v_2(p-1)``) or B, from the low bits of ``n``.
4. For each prime base ``a``, whether ``n`` is an Euler pseudoprime to ``a`` -- i.e.
   ``a**((n-1)/2) == (a/n) (mod n)`` -- computed **rigorously and at scale** as follows.
   Because ``(p-1) | (n-1)`` for every factor ``p``, Fermat gives

       a**((n-1)/2) == 1                     (mod p)   if (n-1)/(p-1) is even,
       a**((n-1)/2) == (a/p) [Legendre]      (mod p)   if (n-1)/(p-1) is odd,

   and ``(n-1)/(p-1)`` is odd iff ``v_2(p-1) == v_2(n-1)``.  The Euler condition holds iff
   this per-prime value equals the global Jacobi symbol ``(a/n) = prod_p (a/p)`` modulo every
   ``p``.  This uses only Fermat and the divisibility ``(p-1)|(n-1)`` -- no assumption about
   the number's class, and no exponentiation modulo the (possibly billion-digit) ``n``.

   When ``n`` is small enough (``--modexp-limit`` bits, default 700000) the literal
   ``gmpy2.powmod(a, (n-1)//2, n)`` is run as an extra gold-standard cross-check.

It reports the largest ``T`` for which the first ``T`` prime bases are *all* Euler liars --
since Euler liars form a subgroup, that means ``n`` is an Euler pseudoprime to every integer
base below the ``(T+1)``-th prime.

Usage:
    python3 verify_euler.py --lambda "2^9*3^5*...*73" n_factors.txt [--max-base 2000] [-j N]
                            [--full-primality] [--modexp-limit BITS]
"""
from __future__ import annotations

import argparse
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np

try:
    import gmpy2
    from gmpy2 import mpz, powmod
    HAVE_GMPY2 = True
except ImportError:
    HAVE_GMPY2 = False

_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

# Per-worker globals (populated by fork inheritance / the pool initializer).
_VALS_OBJ: np.ndarray = np.empty(0, dtype=object)   # exact prime factors
_P_MOD8: np.ndarray = np.empty(0, dtype=np.int8)
_HIGH_IDX: np.ndarray = np.empty(0, dtype=np.int64)  # factors with v_2(p-1) == v_2(n-1)
_HAS_LOW: bool = False


# --------------------------------------------------------------------------- #
def parse_lambda(text: str) -> tuple[int, list[int]]:
    primes: list[int] = []
    value = 1
    for term in text.replace(" ", "").split("*"):
        if not term:
            raise SystemExit("empty factor in Lambda specification")
        m = re.fullmatch(r"(\d+)(?:\^(\d+))?", term)
        if not m:
            raise SystemExit(f"cannot parse factor {term!r}")
        q = int(m.group(1))
        h = int(m.group(2)) if m.group(2) else 1
        if h < 1 or q < 2 or q in primes or any(q % d == 0 for d in range(2, math.isqrt(q) + 1)):
            raise SystemExit(f"Lambda requires distinct prime bases and positive exponents: {term!r}")
        primes.append(q)
        value *= q ** h
    if not primes:
        raise SystemExit("empty Lambda specification")
    return value, sorted(primes)


def read_factors(path: str) -> list[int]:
    out: list[int] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(int(line.split()[0]))
    return out


def _v2(n: int) -> int:
    return (n & -n).bit_length() - 1


def first_primes(t: int) -> list[int]:
    if t <= 0:
        return []
    limit = 15 if t < 6 else int(t * (math.log(t) + math.log(math.log(t)))) + 10
    while True:
        sieve = np.ones(limit + 1, dtype=bool)
        sieve[:2] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                sieve[i * i::i] = False
        primes = np.flatnonzero(sieve).tolist()
        if len(primes) >= t:
            return primes[:t]
        limit *= 2


# ---- primality (strong or rigorous), for validating the factors -------------- #
def _is_strong_prp_py(n: int, a: int) -> bool:
    if n % a == 0:
        return n == a
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    x = pow(a, d, n)
    if x == 1 or x == n - 1:
        return True
    for _ in range(s - 1):
        x = x * x % n
        if x == n - 1:
            return True
    return False


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if HAVE_GMPY2:
        return bool(gmpy2.is_prime(n, 25))
    for p in _MR_BASES:
        if n % p == 0:
            return n == p
    return all(_is_strong_prp_py(n, a) for a in _MR_BASES)


def _proves_prime(n: int, lam_primes: tuple[int, ...]) -> bool:
    if n == 2:
        return True
    if n < 2 or n % 2 == 0:
        return False
    nm1 = n - 1
    qs = [q for q in lam_primes if nm1 % q == 0]
    if HAVE_GMPY2:
        m = mpz(n); e = m - 1; one = mpz(1)
        cof = [e // q for q in qs]
        for a in range(2, 1 << 20):
            am = mpz(a)
            if powmod(am, e, m) != one:
                return False
            if all(powmod(am, c, m) != one for c in cof):
                return True
    else:
        cof = [nm1 // q for q in qs]
        for a in range(2, 1 << 20):
            if pow(a, nm1, n) != 1:
                return False
            if all(pow(a, c, n) != 1 for c in cof):
                return True
    raise RuntimeError(f"primality proof inconclusive for {n}")


_LAM_PRIMES_PRIM: tuple[int, ...] = ()
_FULL_PRIM = False


def _init_prim(lam_primes, full):
    global _LAM_PRIMES_PRIM, _FULL_PRIM
    _LAM_PRIMES_PRIM, _FULL_PRIM = tuple(lam_primes), full


def _prime_chunk(chunk):
    for p in chunk:
        ok = _proves_prime(p, _LAM_PRIMES_PRIM) if _FULL_PRIM else _is_prime(p)
        if not ok:
            return p  # first non-prime found
    return None


# ---- vectorized Legendre bits (reciprocity), self-contained ------------------ #
def _qr_table(q: int) -> np.ndarray:
    t = np.full(q, -1, dtype=np.int8)
    t[0] = 0
    for x in range(1, q):
        t[(x * x) % q] = 1
    return t


def _legendre_bits(base: int, vals_obj: np.ndarray, p_mod8: np.ndarray) -> np.ndarray:
    """uint8 bits: 1 iff Legendre (base/p) == -1, over every factor p (all p > base)."""
    if base == 2:
        return ((p_mod8 == 3) | (p_mod8 == 5)).astype(np.uint8)
    pq = (vals_obj % base).astype(np.int64)
    leg = _qr_table(base)[pq]
    if base % 4 == 3:
        leg = np.where((p_mod8 & 3) == 3, -leg, leg)
    return (leg == -1).astype(np.uint8)


def _init_euler(vals_obj, p_mod8, high_idx, has_low):
    global _VALS_OBJ, _P_MOD8, _HIGH_IDX, _HAS_LOW
    _VALS_OBJ, _P_MOD8, _HIGH_IDX, _HAS_LOW = vals_obj, p_mod8, high_idx, has_low


def _is_euler_liar(base: int) -> bool:
    """Rigorous, scale-free Euler-pseudoprime test for one base (see module docstring)."""
    lb = _legendre_bits(base, _VALS_OBJ, _P_MOD8)     # (a/p) == -1 ?  per factor
    s_bit = int(lb.sum()) & 1                          # (a/n) = prod (a/p), as a bit
    if _HAS_LOW and s_bit != 0:
        # some factor has v_2(p-1) < v_2(n-1): its a**((n-1)/2) == 1 (mod p), so the Euler
        # condition forces (a/n) = +1.  It is not, so a is a witness.
        return False
    if _HIGH_IDX.size:
        # high factors (v_2(p-1) == v_2(n-1)): need (a/p) == (a/n) for each.
        return bool(np.all(lb[_HIGH_IDX] == s_bit))
    return True


def _euler_liar_chunk(bases):
    return [ _is_euler_liar(int(a)) for a in bases ]


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Independent Euler-pseudoprime verifier (scales to millions of factors).")
    ap.add_argument("--lambda", dest="lam", required=True, help="Lambda spec, e.g. 2^9*3^5*...*73")
    ap.add_argument("file", help="prime factors of n (one per line, blank/# ignored)")
    ap.add_argument("--max-base", type=int, default=2000,
                    help="scan the first this-many prime bases to locate the survival boundary")
    ap.add_argument("-j", "--jobs", type=int, default=min(os.cpu_count() or 1, 72))
    ap.add_argument("--full-primality", action="store_true",
                    help="rigorous Lucas/Pocklington proof per factor (default: strong probable-prime)")
    ap.add_argument("--modexp-limit", type=int, default=700_000,
                    help="also run literal a**((n-1)/2) mod n when n has fewer than this many bits")
    args = ap.parse_args(argv)

    lam_value, lam_primes = parse_lambda(args.lam)
    factors = read_factors(args.file)
    K = len(factors)
    mode = "gmpy2" if HAVE_GMPY2 else "pure-Python"
    print(f"Lambda = {lam_value} ({lam_value.bit_length()} bits), primes {lam_primes}")
    print(f"n has {K} prime factors; backend {mode}; jobs {args.jobs}")

    # ---- (1) squarefree + >=3 -------------------------------------------------
    if len(set(factors)) != K:
        print("FAIL: not squarefree (repeated prime factor)")
        return 1
    if K < 3:
        print("FAIL: fewer than 3 prime factors (not a Carmichael number)")
        return 1

    # ---- (2) every factor prime ----------------------------------------------
    chunks = [factors[i:i + 4096] for i in range(0, K, 4096)]
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init_prim,
                             initargs=(lam_primes, args.full_primality)) as ex:
        for bad in ex.map(_prime_chunk, chunks):
            if bad is not None:
                print(f"FAIL: factor {bad} is not prime")
                return 1
    proof = "rigorous Lucas/Pocklington" if args.full_primality else "strong probable-prime"
    if args.full_primality:
        print(f"  [ok] all {K} factors proven prime ({proof})")
    else:
        print(f"  [ok] all {K} factors passed {proof} checks (primality not proven)")

    # ---- (3) Korselt via Lambda: (p-1)|Lambda and n == 1 (mod Lambda) ---------
    for p in factors:
        if lam_value % (p - 1) != 0:
            print(f"FAIL: (p-1) does not divide Lambda for p={p}")
            return 1
    acc = 1
    for p in factors:
        acc = acc * (p % lam_value) % lam_value
    if acc != 1:
        print("FAIL: n != 1 (mod Lambda)  (not a Carmichael number)")
        return 1
    print("  [ok] Carmichael: squarefree, >=3 factors, (p-1)|Lambda, n == 1 (mod Lambda)")

    # ---- class A/B from the low bits of n ------------------------------------
    MASK = (1 << 128) - 1
    n_low = 1
    for p in factors:
        n_low = (n_low * (p & MASK)) & MASK
    v2_n1 = _v2((n_low - 1) & MASK) if n_low != 1 else 128
    v2p = np.fromiter((_v2(p - 1) for p in factors), dtype=np.int64, count=K)
    max_v2p = int(v2p.max())
    klass = "A" if v2_n1 > max_v2p else "B"
    print(f"  v_2(n-1) = {v2_n1}, max_p v_2(p-1) = {max_v2p}  ->  class {klass}")

    # ---- (4) Euler-pseudoprime scan over the first --max-base primes ----------
    vals_obj = np.array(factors, dtype=object)
    p_mod8 = (vals_obj % 8).astype(np.int8)
    high_idx = np.flatnonzero(v2p == v2_n1).astype(np.int64)
    has_low = bool(np.any(v2p < v2_n1))

    bases = first_primes(args.max_base)
    _init_euler(vals_obj, p_mod8, high_idx, has_low)  # also set in-parent for fork inheritance
    liar = [False] * len(bases)
    bchunks = [bases[i:i + 32] for i in range(0, len(bases), 32)]
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init_euler,
                             initargs=(vals_obj, p_mod8, high_idx, has_low)) as ex:
        out = list(ex.map(_euler_liar_chunk, bchunks))
    flat = [b for part in out for b in part]
    # longest run of liars from the start
    survived = 0
    for is_liar in flat:
        if is_liar:
            survived += 1
        else:
            break
    first_fail = bases[survived] if survived < len(bases) else None

    print(f"\n  Euler liar for the first {survived} prime bases "
          f"(largest liar prime base: {bases[survived-1] if survived else 'none'}).")
    if first_fail is not None:
        print(f"  First witness (Euler test detects compositeness) at prime base {first_fail}.")
        print(f"  => n is an Euler pseudoprime to every integer base < {first_fail}.")
    else:
        print(f"  No witness among the first {len(bases)} prime bases "
              f"(increase --max-base to locate the boundary).")

    # ---- gold-standard literal modexp, if n is small enough -------------------
    n_bits = int(sum(p.bit_length() for p in factors))
    if HAVE_GMPY2 and n_bits < args.modexp_limit:
        n = mpz(1)
        for p in factors:
            n *= p
        e = (n - 1) // 2
        check_upto = survived if first_fail is None else survived + 1
        mism = None
        for a in bases[:check_upto]:
            lhs = powmod(a, e, n)
            rhs = gmpy2.jacobi(a, n) % n
            is_liar_modexp = (lhs == rhs)
            if is_liar_modexp != flat[bases.index(a)]:
                mism = a
                break
        if mism is None:
            print(f"  [gold] literal a**((n-1)/2) mod n agrees for all {check_upto} "
                  f"bases checked (n = {n_bits} bits).")
        else:
            print(f"  [gold] MISMATCH at base {mism}: literal modexp disagrees with the "
                  f"scalable test -- investigate!")
            return 1
    elif n_bits >= args.modexp_limit:
        print(f"  (n = {n_bits} bits; skipped literal modexp cross-check, "
              f"--modexp-limit {args.modexp_limit})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
