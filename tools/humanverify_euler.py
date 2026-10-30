#!/usr/bin/env python3
"""Human-auditable verifier that a number is an Euler pseudoprime to many prime bases.

This script is deliberately SIMPLE and SELF-CONTAINED: it imports only ``gmpy2`` and the
standard library, uses plain loops (no numpy, no vectorized reciprocity tricks), and imports
nothing from the construction package. It checks factors with ``gmpy2.is_prime``,
which is a probable-prime screen for large inputs. For a rigorous factor proof,
use ``tools/verify_euler.py --full-primality``. This script's Euler congruence
check is independent of the construction, conditional on the listed factors
actually being prime.

--------------------------------------------------------------------------------------------
The claim.  Given a modulus ``Lambda`` and a file listing the prime factors of a number
``n`` (one prime per line), ``n`` is an **Euler pseudoprime** to base ``a`` iff

        a ** ((n-1)/2)  ==  (a / n)   (mod n)                                   (Euler/S-S)

where ``(a/n)`` is the Jacobi symbol.  A composite ``n`` passing this for base ``a`` fools the
Solovay-Strassen primality test for that base.  We verify it holds for the first ``N`` prime
bases -- and since the Euler liars form a subgroup of ``(Z/nZ)^*``, that means ``n`` fools the
test for EVERY integer base below the ``(N+1)``-th prime.

Why we can check (Euler/S-S) WITHOUT exponentiating modulo the (tens-of-millions-of-digits)
number ``n`` -- four elementary facts, each easy to check by hand:

  (1) Definition:  n is an Euler pseudoprime to a  <=>  a**((n-1)/2) == (a/n) (mod n).
  (2) CRT:  since n is squarefree (distinct prime factors), a congruence holds mod n iff it
      holds modulo every prime factor p | n.  So (1) <=> for every p:  a**((n-1)/2) == (a/n) (mod p).
  (3) Jacobi = product of Legendre:  (a/n) = prod_{p|n} (a/p).
  (4) Fermat:  we verify (p-1) | Lambda and Lambda | (n-1), hence (p-1) | (n-1); therefore
      a**(n-1) == 1 (mod p), so a**((n-1)/2) is a square root of 1 mod p, i.e. +-1, and equals
      a**e (mod p) with the reduced exponent  e = ((n-1)/2) mod (p-1).

  To get e without the giant (n-1)/2:  reduce n modulo 2*Lambda once (a ~125-bit number R).
  Because every (p-1) divides Lambda, 2(p-1) divides 2*Lambda, so
        (n-1)/2 mod (p-1)  =  ((R - 1) mod 2(p-1)) // 2 .

So per prime p we compute  s_p = a**e_p mod p  (a +-1), and n is an Euler pseudoprime to a iff
every ``s_p`` equals the global Jacobi symbol ``(a/n)``.  That is the whole verification.

As an extra anchor, when ``n`` is small enough (``--modexp-limit`` bits) the script ALSO forms
``n`` and checks the raw definition ``a**((n-1)//2) mod n == (a/n)`` literally -- no reduction
at all -- and confirms the per-prime method agrees.  (Run it on the small committed example
``tests/fixtures/euler_golden.txt`` to watch both methods agree; that anchors the per-prime
method for the huge headline number, where the literal exponentiation is infeasible.)

Usage:
    python3 humanverify_euler.py --lambda "2^9*3^5*...*73" factors.txt [--max-base N] [-j J]
                                 [--modexp-limit BITS]
"""
from __future__ import annotations

import argparse
import math
import os
import re
from multiprocessing import Pool

import gmpy2
from gmpy2 import jacobi, mpz, powmod


# --------------------------------------------------------------------------- #
def parse_lambda(text: str) -> tuple[int, list[int]]:
    """(Lambda value, sorted list of its prime factors) from a spec like '2^9*3^5*...'."""
    value, primes = 1, []
    for term in text.replace(" ", "").split("*"):
        if not term:
            raise SystemExit("empty factor in Lambda specification")
        m = re.fullmatch(r"(\d+)(?:\^(\d+))?", term)
        if not m:
            raise SystemExit(f"cannot parse factor {term!r}")
        q = int(m.group(1)); h = int(m.group(2)) if m.group(2) else 1
        if h < 1 or q < 2 or q in primes or any(q % d == 0 for d in range(2, math.isqrt(q) + 1)):
            raise SystemExit(f"Lambda requires distinct prime bases and positive exponents: {term!r}")
        value *= q ** h; primes.append(q)
    if not primes:
        raise SystemExit("empty Lambda specification")
    return value, sorted(primes)


def read_factors(path: str) -> list[int]:
    """The prime factors of n, one per line ('#' comments and blank lines ignored)."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(int(line.split()[0]))
    return out


def first_primes(t: int) -> list[int]:
    """The first t primes, by a plain sieve."""
    if t <= 0:
        return []
    limit = 15 if t < 6 else int(t * (math.log(t) + math.log(math.log(t)))) + 10
    while True:
        sieve = bytearray([1]) * (limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
        primes = [i for i in range(limit + 1) if sieve[i]]
        if len(primes) >= t:
            return primes[:t]
        limit *= 2


# ---- per-base Euler test (the four facts above), run in worker processes ----- #
# Filled in each worker by the pool initializer (inherited by fork; also passed explicitly).
_FACTORS: list[int] = []       # the prime factors p of n
_EXP: list[int] = []           # e_p = ((n-1)/2) mod (p-1), precomputed once


def _init(factors, exps):
    global _FACTORS, _EXP
    _FACTORS, _EXP = factors, exps


def _is_euler_liar(a: int) -> bool:
    """True iff a**((n-1)/2) == (a/n) (mod n), checked prime-by-prime (facts 2-4)."""
    jac = 1                                   # (a/n) = product of Legendre symbols (a/p)
    per_prime = []                            # s_p = a**((n-1)/2) mod p, as +-1
    for p, e in zip(_FACTORS, _EXP):
        jac *= jacobi(a, p)                   # (a/p); p is prime so this is the Legendre symbol
        r = powmod(a, e, p)                   # a**((n-1)/2) mod p  (r is a square root of 1)
        if r == 1:
            per_prime.append(1)
        elif r == p - 1:
            per_prime.append(-1)
        else:                                 # cannot happen if (p-1)|(n-1); guard anyway
            return False
    # Euler condition: a**((n-1)/2) == (a/n) modulo every p, i.e. every s_p equals (a/n).
    return all(s == jac for s in per_prime)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Human-auditable Euler-pseudoprime verifier.")
    ap.add_argument("--lambda", dest="lam", required=True, help="Lambda spec, e.g. 2^9*3^5*...")
    ap.add_argument("file", help="prime factors of n, one per line")
    ap.add_argument("--max-base", type=int, default=0,
                    help="check the first this-many prime bases (default: all bases below the "
                         "smallest prime factor of n, the most that can be claimed)")
    ap.add_argument("-j", "--jobs", type=int, default=min(os.cpu_count() or 1, 72))
    ap.add_argument("--modexp-limit", type=int, default=40_000,
                    help="also do the literal a**((n-1)//2) mod n check when n has fewer than "
                         "this many bits (anchors the per-prime method against the raw definition)")
    args = ap.parse_args(argv)

    lam_value, lam_primes = parse_lambda(args.lam)
    factors = read_factors(args.file)
    k = len(factors)
    print(f"Lambda = {lam_value} ({lam_value.bit_length()} bits)")
    print(f"n = product of {k} listed factors (screened with gmpy2.is_prime)")

    # ---- (A) n is a Carmichael number: squarefree, >=3 factors, and Korselt via Lambda ----
    if len(set(factors)) != k:
        print("FAIL: a prime factor is repeated (n is not squarefree)"); return 1
    if k < 3:
        print("FAIL: fewer than 3 prime factors"); return 1
    for p in factors:
        if not gmpy2.is_prime(p):
            print(f"FAIL: factor {p} is not prime"); return 1
        if lam_value % (p - 1) != 0:
            print(f"FAIL: (p-1) does not divide Lambda for p = {p}"); return 1
    n_mod_lam = 1
    for p in factors:
        n_mod_lam = n_mod_lam * (p % lam_value) % lam_value
    if n_mod_lam != 1:
        print("FAIL: n != 1 (mod Lambda) -- not a Carmichael number"); return 1
    print(f"  [ok] Carmichael conditions pass for {k} distinct factors screened by "
          f"gmpy2.is_prime; full factor primality is not proven here. "
          f"Each (p-1)|Lambda and n == 1 (mod Lambda)")
    # (p-1)|Lambda and Lambda|(n-1)  =>  (p-1)|(n-1) for every p (used by fact 4).

    # ---- reduced exponents e_p = ((n-1)/2) mod (p-1), from R = n mod 2*Lambda ----
    two_lam = 2 * lam_value
    R = 1
    for p in factors:
        R = R * (p % two_lam) % two_lam          # R = n mod 2*Lambda (a ~125-bit number)
    exps = [((R - 1) % (2 * (p - 1))) // 2 for p in factors]

    # ---- (B) Euler pseudoprime to the first N prime bases ----
    bases = first_primes(args.max_base) if args.max_base > 0 else first_primes(0)
    if not bases:
        # default: every prime below the smallest factor is coprime to n and safe to test
        smallest = min(factors)
        bases = [b for b in first_primes(2000) if b < smallest]
    with Pool(processes=args.jobs, initializer=_init, initargs=(factors, exps)) as pool:
        verdicts = pool.map(_is_euler_liar, bases, chunksize=8)
    survived = 0
    for ok in verdicts:
        if ok:
            survived += 1
        else:
            break
    largest = bases[survived - 1] if survived else None
    first_witness = bases[survived] if survived < len(bases) else None
    print(f"  [ok] Euler pseudoprime to the first {survived} prime bases "
          f"(largest liar prime base: {largest}).")
    if first_witness is not None:
        print(f"       first witness at prime base {first_witness}  ->  n is an Euler pseudoprime "
              f"to every integer base < {first_witness}.")
    else:
        print(f"       no witness among the {len(bases)} bases tested "
              f"(raise --max-base to find the boundary).")

    # ---- (C) literal-definition anchor for small n ----
    n_bits = sum(p.bit_length() for p in factors)
    checked = survived + (1 if first_witness is not None else 0)
    if n_bits < args.modexp_limit:
        n = mpz(1)
        for p in factors:
            n *= p
        half = (n - 1) // 2
        for a in bases[:checked]:
            literal_liar = (powmod(a, half, n) == jacobi(a, n) % n)   # the raw definition
            if literal_liar != verdicts[bases.index(a)]:
                print(f"  [gold] MISMATCH at base {a}: literal a**((n-1)//2) mod n disagrees "
                      f"with the per-prime method -- investigate!"); return 1
        print(f"  [gold] literal a**((n-1)//2) mod n agrees with the per-prime method for all "
              f"{checked} bases checked (n = {n_bits} bits).")
    else:
        print(f"  (n = {n_bits} bits: literal modexp skipped; per-prime method above is exact.)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
