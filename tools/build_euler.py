#!/usr/bin/env python3
"""Driver: build an Euler pseudoprime to many prime bases from a flat Lambda.

Pipeline: generate P (with a factor floor and a class-A strategy) -> push-down
(coprime base Carmichaels) -> F2 Jacobi-kernel construction (pdc.euler) -> self-check ->
write the prime factors of n to --out for the standalone verifier tools/verify_euler.py.

Examples:
    # fixed target: an Euler pseudoprime to the first 100 primes, from the record-flat Lambda
    python3 tools/build_euler.py --preset record-flat --bases 100 --out epsp100.txt

    # maximize: largest first-t-primes Euler pseudoprime with all factors above a floor
    python3 tools/build_euler.py --preset record-flat --maximize --floor 4000 --out epsp_max.txt
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdc.lam import Lambda, preset, PRESETS
from pdc.primes import generate_prime_set
from pdc.pushdown import push_down
from pdc.euler import (construct_euler_pseudoprime, construct_maximal, verify_construction,
                       structural_check, first_primes)


def _primes_below(bound: int) -> int:
    """Count of primes strictly below ``bound`` (base_cap for --maximize)."""
    import numpy as np
    if bound <= 2:
        return 0
    sieve = np.ones(bound, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(bound ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = False
    return int(sieve.sum())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Construct an Euler pseudoprime to many bases.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--preset", choices=sorted(PRESETS), help="named Lambda preset")
    g.add_argument("--lambda", dest="lam", help="explicit Lambda spec, e.g. 2^9*3^5*...*73")

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--bases", type=int, help="target the first this-many prime bases")
    mode.add_argument("--maximize", action="store_true",
                      help="find the largest first-t-primes pseudoprime for the chosen floor")

    ap.add_argument("--floor", type=int, default=0,
                    help="min prime factor (default: just above the largest target base). "
                         "For --maximize this sets the base_cap = #primes below the floor.")
    ap.add_argument("--want-sparse", action="store_true",
                    help="prefer a lower-weight kernel vector (smaller n)")
    ap.add_argument("--class-a-method", choices=("filter", "pushdown", "pushdown-4q"),
                    default="filter",
                    help="force class A by filtering maximal-v2 primes (existing method) "
                         "or by retaining the full pool and using 2Q (pushdown) or 4Q "
                         "(pushdown-4q) at the 2-primary stage")
    ap.add_argument("-j", "--jobs", type=int, default=min(os.cpu_count() or 1, 72))
    ap.add_argument("--out", help="write the prime factors of n here (one per line)")
    args = ap.parse_args(argv)

    lam = preset(args.preset) if args.preset else Lambda.parse(args.lam)

    if args.maximize:
        if args.floor <= 0:
            ap.error("--maximize requires --floor (the minimum prime factor)")
        floor = args.floor
        base_cap = _primes_below(floor)
    else:
        largest_base = first_primes(args.bases)[-1]
        floor = args.floor if args.floor > 0 else largest_base + 1
        if floor <= largest_base:
            ap.error(f"--floor {floor} must exceed the largest target base {largest_base}")
        base_cap = None

    print(f"Lambda = {lam}")
    print(f"  value = {lam.value} ({lam.value.bit_length()} bits)")
    print(f"  factor floor (min prime) = {floor}; "
          f"class-A method = {args.class_a_method}; jobs = {args.jobs}")

    t0 = time.time()
    multiplier = {"filter": 1, "pushdown": 2, "pushdown-4q": 4}[args.class_a_method]
    P = generate_prime_set(
        lam, jobs=args.jobs, min_prime=floor,
        class_a=(args.class_a_method == "filter"),
        two_adic_multiplier=multiplier)
    t1 = time.time()
    pool_label = "maximal-v2 filtered" if args.class_a_method == "filter" else "full"
    print(f"|P| = {P.N} primes ({pool_label} pool)  [{t1 - t0:.1f}s]")
    print(f"  push-down moduli = {P.stage_moduli}")
    two_col = lam.primes.index(2)
    base2_mode = ("automatic (outputs are 1 mod 8)"
                  if P.stage_moduli[two_col] % 8 == 0 else "linear-algebra row")
    print(f"  base 2 = {base2_mode}")

    res = push_down(P, log=lambda s: None)
    t2 = time.time()
    ncols = len(res.base)
    print(f"push-down: {ncols} base Carmichaels (columns)  [{t2 - t1:.1f}s]")

    if args.maximize:
        print(f"maximizing over the first {base_cap} primes (all < floor {floor}) ...")
        eres = construct_maximal(res, base_cap, want_sparse=args.want_sparse,
                                 jobs=args.jobs, log=print)
    else:
        eres = construct_euler_pseudoprime(res, args.bases, want_sparse=args.want_sparse,
                                           jobs=args.jobs, log=print)
    t3 = time.time()

    if not eres.solved:
        print(f"\nNO SOLUTION: the kernel is trivial "
              f"(rank {eres.rank} == columns {eres.n_columns}); "
              f"need more base Carmichaels (raise |P| / lower the floor) or fewer bases.")
        return 1

    t = len(eres.target_bases)
    n_bits = sum(p.bit_length() for p in eres.factor_values)
    print(f"\nSOLVED: Euler pseudoprime to the first {t} primes "
          f"(largest base {eres.target_bases[-1]}).")
    print(f"  {len(eres.auto_bases)} bases divide Lambda (free); "
          f"{len(eres.hard_bases)} constrained; rank(M) = {eres.rank}; columns = {eres.n_columns}")
    print(f"  selected {len(eres.selected)} base Carmichaels -> n has {eres.num_factors} "
          f"prime factors (~{n_bits} bits)  [construct {t3 - t2:.1f}s]")

    # Write the artifact FIRST (before any verification), so the expensive part is never lost.
    if args.out:
        with open(args.out, "w") as f:
            f.write(f"# Euler pseudoprime to the first {t} primes (largest base "
                    f"{eres.target_bases[-1]})\n")
            f.write(f"# Lambda = {lam}\n")
            f.write(f"# class-A method = {args.class_a_method}; "
                    f"push-down moduli = {P.stage_moduli}\n")
            f.write(f"# n = product of the {eres.num_factors} primes below (~{n_bits} bits)\n")
            for p in eres.factor_values:
                f.write(f"{p}\n")
        print(f"  wrote {eres.num_factors} prime factors to {args.out}  [{time.time() - t3:.1f}s]")

    # Cheap structural self-check (one pass over the factors): is n a class-A Carmichael?
    sc = structural_check(eres)
    print(f"\n  structural self-check: squarefree={sc.squarefree} >=3factors={sc.at_least_3_factors} "
          f"n==1(mod Lambda)={sc.n_mod_lambda_is_one}")
    print(f"    v_2(n-1)={sc.v2_n_minus_1} > max_p v_2(p-1)={sc.max_v2_p}  ->  class A = {sc.class_a}")

    # Full per-base Euler check inline only when cheap; otherwise defer to the parallel verifier.
    if eres.num_factors * t <= 300_000_000:
        chk = verify_construction(eres)
        print(f"  base self-check: survived first {chk.survived_bases} prime bases; "
              f"first non-liar: {chk.failing_base}; overall ok={chk.ok}")
        ok = chk.ok
    else:
        print(f"  (base self-check deferred: {eres.num_factors} factors x {t} bases is large)")
        ok = sc.is_class_a_carmichael

    if args.out:
        print(f"\n  verify independently (parallel, construction-independent):\n"
              f"    python3 tools/verify_euler.py --lambda \"{lam}\" {args.out} --max-base {t}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
