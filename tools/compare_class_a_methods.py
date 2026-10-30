#!/usr/bin/env python3
"""Small deterministic comparison of three class-A Euler construction paths.

This is deliberately a demo-scale research tool, not a paper-scale benchmark. It compares
the existing maximal-2-adic-valuation prime filter with strengthened-modulus alternatives
that keep the complete ``p-1 | Lambda`` pool and use ``2Q`` or ``4Q`` at the 2-primary
push-down stage.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdc.euler import (construct_euler_pseudoprime, construct_maximal, first_primes,
                       structural_check, verify_construction)
from pdc.lam import Lambda, PRESETS, preset
from pdc.primes import _v2, generate_prime_set
from pdc.pushdown import push_down
from pdc.verify import verify_base_numbers


@dataclass
class Trial:
    name: str
    pool: int
    two_modulus: int
    base_columns: int
    thrown_objects: int
    thrown_factors: int
    rank: int
    nullity: int
    solved: bool
    target_bases: int
    largest_base: int
    first_excluded_base: int | None
    hard_bases: int
    selected_columns: int
    result_factors: int
    class_a: bool
    result_verified: bool
    bases_verified: bool
    generate_seconds: float
    pushdown_seconds: float
    linear_seconds: float
    verify_seconds: float


def _method_kwargs(name: str) -> dict:
    if name == "filter":
        return {"class_a": True}
    if name == "2Q":
        return {"two_adic_multiplier": 2}
    if name == "4Q":
        return {"two_adic_multiplier": 4}
    raise ValueError(f"unknown class-A method: {name}")


def _run(name: str, lam: Lambda, floor: int, bases: int, jobs: int,
         want_sparse: bool, maximize: bool) -> tuple[Trial, object]:
    kwargs = _method_kwargs(name)
    t0 = time.perf_counter()
    primes = generate_prime_set(lam, jobs=jobs, min_prime=floor, **kwargs)
    t1 = time.perf_counter()
    result = push_down(primes, jobs=jobs)
    t2 = time.perf_counter()
    if maximize:
        euler = construct_maximal(
            result, bases, want_sparse=want_sparse, jobs=jobs)
    else:
        euler = construct_euler_pseudoprime(
            result, bases, want_sparse=want_sparse, jobs=jobs)
    t3 = time.perf_counter()

    carmichael, _ = verify_base_numbers(result, jobs=jobs)
    structural = structural_check(euler)
    # This comparison requires the construction-independent factorwise Euler/Jacobi pass.
    # Literal giant-integer modular exponentiation is covered by the GMP-backed test suite.
    euler_check = verify_construction(euler, gold_modexp_bit_limit=0)
    t4 = time.perf_counter()
    two_col = lam.primes.index(2)
    first_excluded = None
    if maximize and len(euler.target_bases) < bases:
        first_excluded = first_primes(len(euler.target_bases) + 1)[-1]
    trial = Trial(
        name=name,
        pool=primes.N,
        two_modulus=primes.stage_moduli[two_col],
        base_columns=len(result.base),
        thrown_objects=sum(s.thrown for s in result.stages),
        thrown_factors=sum(s.thrown_factors for s in result.stages),
        rank=euler.rank,
        nullity=euler.n_columns - euler.rank,
        solved=euler.solved,
        target_bases=len(euler.target_bases),
        largest_base=euler.target_bases[-1] if euler.target_bases else 0,
        first_excluded_base=first_excluded,
        hard_bases=len(euler.hard_bases),
        selected_columns=len(euler.selected),
        result_factors=euler.num_factors,
        class_a=structural.class_a,
        result_verified=euler_check.ok,
        bases_verified=all(c.ok for c in carmichael),
        generate_seconds=t1 - t0,
        pushdown_seconds=t2 - t1,
        linear_seconds=t3 - t2,
        verify_seconds=t4 - t3,
    )
    return trial, primes


def _show(trials: list[Trial]) -> None:
    rows = [
        ("admissible primes", [t.pool for t in trials]),
        ("2-primary stage modulus", [t.two_modulus for t in trials]),
        ("base columns after push-down", [t.base_columns for t in trials]),
        ("discarded state objects", [t.thrown_objects for t in trials]),
        ("discarded underlying primes", [t.thrown_factors for t in trials]),
        ("linear rank", [t.rank for t in trials]),
        ("linear nullity", [t.nullity for t in trials]),
        ("Euler system solved", [t.solved for t in trials]),
        ("consecutive prime bases reached", [t.target_bases for t in trials]),
        ("largest prime base reached", [t.largest_base for t in trials]),
        ("first excluded prime base", [t.first_excluded_base for t in trials]),
        ("hard-base constraints included", [t.hard_bases for t in trials]),
        ("selected base columns", [t.selected_columns for t in trials]),
        ("prime factors in selected result", [t.result_factors for t in trials]),
        ("selected result is class A", [t.class_a for t in trials]),
        ("selected result passes all target bases", [t.result_verified for t in trials]),
        ("base Carmichaels structurally verified", [t.bases_verified for t in trials]),
        ("prime generation seconds", [f"{t.generate_seconds:.4f}" for t in trials]),
        ("push-down seconds", [f"{t.pushdown_seconds:.4f}" for t in trials]),
        ("linear algebra seconds", [f"{t.linear_seconds:.4f}" for t in trials]),
        ("verification seconds", [f"{t.verify_seconds:.4f}" for t in trials]),
    ]
    width = max(len(label) for label, _ in rows)
    print(f"{'metric':<{width}}" + "".join(f"  {t.name:>12}" for t in trials))
    print(f"{'-' * width}" + "".join(f"  {'-' * 12}" for _ in trials))
    for label, values in rows:
        print(f"{label:<{width}}" + "".join(f"  {str(value):>12}" for value in values))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Compare class A via prime filtering against class A via 2Q push-down.")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--preset", choices=sorted(PRESETS), default="demo")
    group.add_argument("--lambda", dest="lam", help="explicit factored Lambda")
    ap.add_argument("--bases", type=int, default=20,
                    help="number of consecutive prime bases (default: 20)")
    ap.add_argument("--floor", type=int,
                    help="minimum prime factor (default: one above the largest base)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="worker count; default 1 keeps this comparison small")
    ap.add_argument("--want-sparse", action="store_true")
    ap.add_argument("--maximize", action="store_true",
                    help="find the maximal consecutive prefix up to --bases")
    ap.add_argument("--json-out", help="write complete machine-readable trial records")
    args = ap.parse_args(argv)

    if args.bases < 1:
        ap.error("--bases must be positive")
    lam = Lambda.parse(args.lam) if args.lam else preset(args.preset)
    largest_base = first_primes(args.bases)[-1]
    floor = args.floor if args.floor is not None else largest_base + 1
    if floor <= largest_base:
        ap.error(f"--floor {floor} must exceed largest target base {largest_base}")

    print(f"Lambda = {lam}")
    target_word = "cap" if args.maximize else "target"
    print(f"{target_word} = first {args.bases} prime bases; "
          f"factor floor = {floor}; jobs = {args.jobs}")
    runs = [_run(name, lam, floor, args.bases, args.jobs, args.want_sparse, args.maximize)
            for name in ("filter", "2Q", "4Q")]
    trials = [trial for trial, _ in runs]
    filtered, two_q, four_q = trials
    filtered_pool, two_q_pool, four_q_pool = [pool for _, pool in runs]

    filtered_values = set(filtered_pool.values)
    full_values = set(two_q_pool.values)
    if not filtered_values <= full_values:
        raise AssertionError("filtered pool is not a subset of the full pool")
    if set(four_q_pool.values) != full_values:
        raise AssertionError("2Q and 4Q did not preserve the same admissible-prime pool")
    recovered = full_values - filtered_values
    s = _v2(lam.value)
    if any(_v2(p - 1) != s for p in recovered):
        raise AssertionError("the recovered pool contains a prime not removed by the old v2 filter")

    print()
    _show(trials)
    pct = 100.0 * len(recovered) / len(filtered_values) if filtered_values else float("inf")
    print(f"\nrecovered primes: {len(recovered)} ({pct:.2f}% of the filtered-pool size)")
    print("timings are exploratory single runs, not benchmarks")

    if args.json_out:
        record = {
            "lambda": str(lam),
            "lambda_value": lam.value,
            "factor_floor": floor,
            "base_mode": "maximal_prefix" if args.maximize else "fixed_target",
            "base_target_or_cap": args.bases,
            "worker_count": args.jobs,
            "want_sparse": args.want_sparse,
            "timing_status": "diagnostics from single runs; not benchmarks",
            "methods": [asdict(trial) for trial in trials],
        }
        with open(args.json_out, "w") as f:
            json.dump(record, f, indent=2, sort_keys=True)
            f.write("\n")

    ok = all(t.solved and t.class_a and t.result_verified and t.bases_verified
             for t in trials)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
