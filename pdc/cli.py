"""Command-line driver: Lambda -> P -> push-down -> verify -> report."""

from __future__ import annotations

import argparse
import sys
import time

from .lam import Lambda, DEFAULT_LAMBDA, PRESETS, preset
from .primes import generate_prime_set, HAVE_GMPY2
from .pushdown import push_down
from .verify import verify_base_numbers, read_factored_carmichaels, is_carmichael
from .coverage import reachable_counts, achievable_ks, format_ranges, base_histogram


def _run(args) -> int:
    if args.preset:
        lam = preset(args.preset)
    elif args.lam:
        lam = Lambda.parse(args.lam)
    else:
        lam = DEFAULT_LAMBDA
    quiet = args.quiet
    say = (lambda *a: None) if quiet else (lambda *a: print(*a))

    say(f"Lambda = {lam}")
    say(f"       = {lam.value}")
    say(f"r = {lam.r} prime powers; Q_i (desc) = {list(lam.prime_powers)}")
    say(f"K(Lambda) ~ {lam.k_estimate()}  (Loh-Niebuhr estimate of |P|)")
    say(f"primality backend: {'gmpy2' if HAVE_GMPY2 else 'pure-Python'}; jobs = {args.jobs or 'auto'}")

    t0 = time.time()
    progress = None if quiet else (
        lambda done, total, found: print(
            f"  ...{done}/{total} subtrees, |P|={found}", end="\r", file=sys.stderr))
    P = generate_prime_set(lam, jobs=args.jobs, progress=progress)
    say(f"\n|P| = {P.N} primes generated and proven in {time.time()-t0:.2f}s")

    t1 = time.time()
    result = push_down(P, log=(None if quiet else say), jobs=args.jobs)
    say(f"Push-down finished in {time.time()-t1:.2f}s")

    if args.skip_base_verify:
        say(f"\nFound {len(result.base)} base numbers "
            f"(per-base verification SKIPPED via --skip-base-verify; "
            f"the reachable-k coverage below is therefore unverified).")
    else:
        t2 = time.time()
        carm, other = verify_base_numbers(result, jobs=args.jobs)
        say(f"\nFound {len(result.base)} base numbers "
            f"({len(carm)} Carmichael, {len(other)} with <3 factors) "
            f"[verified in {time.time()-t2:.2f}s].")

        bad = [c for c in carm if not c.ok]
        if bad:
            print("VERIFICATION FAILED for:")
            for c in bad:
                print(f"  n={c.n} factors={c.factors}: {c.reason}")
            return 1
        say("All base numbers verified == 1 (mod Lambda); all >=3-factor numbers pass Korselt.")

    hist = base_histogram(result.factor_counts)
    say("\nBase number of factors (count in parentheses if >1):")
    say("  " + " ".join(
        f"{k}({v})" if v > 1 else f"{k}" for k, v in sorted(hist.items())))

    reach = reachable_counts(result.factor_counts)   # single DP pass
    ks = achievable_ks(reach)                         # O(top) bit extraction
    say(f"\nCarmichael numbers exist with k prime factors for k in:")
    say("  " + format_ranges(ks))
    say(f"\n(total distinct achievable k>=3: {len(ks)}; "
        f"largest k = {reach.bit_length()-1})")

    if args.write_carmichaels:
        with open(args.write_carmichaels, "w") as f:
            for indices in result.base:
                for j in indices:
                    f.write(f"{P.values[j]}\n")
                f.write("\n")
        say(f"\nWrote base Carmichaels (factored) to {args.write_carmichaels}")
    return 0


def _verify_file(args) -> int:
    groups = read_factored_carmichaels(args.verify_file)
    ok = 0
    for g in groups:
        chk = is_carmichael(g)
        if chk.ok:
            ok += 1
        else:
            print(f"NOT Carmichael ({chk.reason}): {g[:3]}... ({len(g)} factors)")
    print(f"{ok}/{len(groups)} groups verified as Carmichael numbers.")
    return 0 if ok == len(groups) else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="pdc",
        description="Algorithm 1 (Many Carmichaels subset-product) -- modern re-implementation.")
    p.add_argument("--lambda", dest="lam", metavar="SPEC",
                   help='Lambda as e.g. "2^7*3^4*5^2*7^2*11*13*17*19*23" (default: a small demo Lambda).')
    p.add_argument("--preset", metavar="NAME", choices=sorted(PRESETS),
                   help=f"use a named Lambda preset instead of --lambda. Choices: {', '.join(sorted(PRESETS))}.")
    p.add_argument("--jobs", type=int, metavar="N", default=None,
                   help="parallel worker processes for prime generation and base-number "
                        "verification (default: min(cpu_count, 72); 1 = serial).")
    p.add_argument("--skip-base-verify", action="store_true",
                   help="skip per-base Carmichael verification -- for fast search runs. The "
                        "reachable-k coverage is then reported but UNVERIFIED (verify the "
                        "winning witness separately).")
    p.add_argument("--write-carmichaels", metavar="FILE",
                   help="write the base Carmichaels in factored form to FILE.")
    p.add_argument("--verify-file", metavar="FILE",
                   help="instead of constructing, verify a factored-Carmichael file (e.g. carmichaels.golden).")
    p.add_argument("--quiet", action="store_true", help="suppress progress output.")
    args = p.parse_args(argv)

    if args.verify_file:
        return _verify_file(args)
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
