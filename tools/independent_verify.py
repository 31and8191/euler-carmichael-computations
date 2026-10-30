#!/usr/bin/env python3
"""Independent Carmichael verifier for a factored-base file at any scale.

Deliberately self-contained: it imports nothing from `pdc`, so it does not trust the
construction code in any way.  Given a modulus Lambda and a file of base numbers in
factored form (one prime per line, blank line between numbers -- the format written by
`python3 -m pdc --write-carmichaels`), it confirms each base is a Carmichael number.

It does NOT form the giant integer n (a base can have millions of factors, so n would have
hundreds of millions of digits).  Instead it uses the Korselt-via-Lambda argument, which
is exact and runs in O(#factors):

    n is Carmichael  <=  n is squarefree (distinct primes), has >= 3 prime factors,
                         and (p-1) | (n-1) for every prime p | n.

Because every factor satisfies (p-1) | Lambda (checked here), and n == 1 (mod Lambda)
(checked here, by multiplying the factors' residues modulo Lambda), we get
(p-1) | Lambda | (n-1) for free -- Korselt's divisibility, with no big integer.

Primality of each factor:
  * default: a strong test -- gmpy2.is_prime when gmpy2 is installed (BPSW + extra MR),
    else deterministic Miller-Rabin (a proof below ~2**81; a strong probable-prime test
    above, which only the largest factors of a big Lambda reach).
  * --full-primality: a rigorous proof for every factor via the converse of Fermat
    (Lucas/Pocklington).  This is valid here because p-1 divides the supplied Lambda and
    is therefore fully factored -- its prime divisors are a subset of Lambda's primes.

Work is parallelised across processes (one base per task).  gmpy2 and multiprocessing are
standard, construction-independent tools; the verification logic above is unchanged.

Usage:
    python3 independent_verify.py --lambda "2^9*3^5*5^3*...*73" bases.txt [-j N] [--full-primality]
"""
from __future__ import annotations

import argparse
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor

try:
    import gmpy2
    from gmpy2 import mpz, powmod
    HAVE_GMPY2 = True
except ImportError:
    HAVE_GMPY2 = False

_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

# Per-worker globals (set by the pool initializer).
_LAM = 1
_LAM_PRIMES: tuple[int, ...] = ()
_FULL = False


def parse_lambda(text: str) -> tuple[int, list[int]]:
    """Return (Lambda value, sorted list of its prime factors) from a spec like 2^9*3^5*..."""
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


def _is_strong_prp_py(n: int, a: int) -> bool:
    if n % a == 0:
        return n == a
    d = n - 1
    s = 0
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


def _proves_prime(n: int) -> bool:
    """Rigorous Lucas/Pocklington proof: n-1 | Lambda, so its prime divisors are known."""
    if n == 2:
        return True
    if n < 2 or n % 2 == 0:
        return False
    nm1 = n - 1
    qs = [q for q in _LAM_PRIMES if nm1 % q == 0]
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


def _init(lam_value, lam_primes, full):
    global _LAM, _LAM_PRIMES, _FULL
    _LAM, _LAM_PRIMES, _FULL = lam_value, tuple(lam_primes), full


def _verify_group(factors):
    """Return (status, detail): status in {'carmichael','block','fail'}."""
    K = len(factors)
    if len(set(factors)) != K:
        return ("fail", "not squarefree (repeated prime)")
    prime_ok = _proves_prime if _FULL else _is_prime
    acc = 1
    for p in factors:
        if not prime_ok(p):
            return ("fail", f"factor {p} is not prime")
        if _LAM % (p - 1) != 0:
            return ("fail", f"(p-1) does not divide Lambda for p={p}")
        acc = acc * (p % _LAM) % _LAM
    if acc != 1:
        return ("fail", "n != 1 (mod Lambda)")
    if K < 3:
        return ("block", f"{K} factors")
    return ("carmichael", "")


def read_groups(path: str):
    group: list[int] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                if group:
                    yield group
                    group = []
                continue
            group.append(int(line.split()[0]))
    if group:
        yield group


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Independent Carmichael verifier (scales to millions of factors).")
    ap.add_argument("--lambda", dest="lam", required=True, help="Lambda spec, e.g. 2^9*3^5*...")
    ap.add_argument("file", help="factored-base file (one prime per line, blank-separated)")
    ap.add_argument("-j", "--jobs", type=int, default=min(os.cpu_count() or 1, 72))
    ap.add_argument("--full-primality", action="store_true",
                    help="rigorous Lucas/Pocklington proof per factor (default: strong MR/BPSW)")
    args = ap.parse_args(argv)

    lam_value, lam_primes = parse_lambda(args.lam)
    mode = "gmpy2" if HAVE_GMPY2 else "pure-Python"
    proof = "rigorous Lucas/Pocklington" if args.full_primality else "strong probable-prime"
    print(f"Lambda = {lam_value}  ({lam_value.bit_length()} bits), primes {lam_primes}")
    print(f"backend: {mode}; primality: {proof}; jobs: {args.jobs}")

    groups = list(read_groups(args.file))
    n_carm = n_block = 0
    failures = []
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init,
                             initargs=(lam_value, lam_primes, args.full_primality)) as ex:
        for i, (status, detail) in enumerate(ex.map(_verify_group, groups, chunksize=64)):
            if status == "carmichael":
                n_carm += 1
            elif status == "block":
                n_block += 1
            else:
                failures.append((i, detail))

    status = "proven Carmichael" if args.full_primality else "Carmichael candidates (factor probable-prime checks)"
    print(f"\n{len(groups)} base numbers: {n_carm} {status}, "
          f"{n_block} building blocks (<3 factors), {len(failures)} FAILURES.")
    for i, d in failures[:20]:
        print(f"  FAIL base #{i}: {d}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
