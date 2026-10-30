"""End-to-end: push-down output is genuinely Carmichael, and coverage is correct."""

from itertools import combinations

from pdc.lam import Lambda
from pdc.primes import generate_prime_set
from pdc.pushdown import push_down
from pdc.verify import verify_base_numbers, is_carmichael
from pdc.coverage import reachable_set, reachable_counts


def _run(spec):
    lam = Lambda.parse(spec)
    P = generate_prime_set(lam)
    return lam, P, push_down(P)


def test_every_base_number_is_one_mod_lambda_and_carmichael():
    lam, P, result = _run("2^6 * 3^3 * 5^2 * 7^2 * 11 * 13 * 17 * 19")
    L = lam.value
    assert result.base, "no base numbers produced"
    seen_primes = set()
    for indices in result.base:
        factors = [P.values[j] for j in indices]
        # disjoint prime usage across all base numbers (=> squarefree products)
        assert seen_primes.isdisjoint(indices)
        seen_primes.update(indices)
        n = 1
        for p in factors:
            n *= p
        assert n % L == 1
        if len(factors) >= 3:
            assert is_carmichael(factors).ok


def test_verifier_flags_none():
    _, _, result = _run("2^5 * 3^3 * 5^2 * 7 * 11 * 13 * 17")
    carm, _ = verify_base_numbers(result)
    assert all(c.ok for c in carm)


def test_coverage_matches_bruteforce_subset_sums():
    _, _, result = _run("2^5 * 3^3 * 5^2 * 7 * 11 * 13")
    counts = result.factor_counts
    # brute-force subset sums (feasible: few base numbers)
    sums = {0}
    for f in counts:
        sums |= {s + f for s in sums}
    expected = sorted(k for k in sums if k >= 3)
    assert reachable_set(counts) == expected
    reach = reachable_counts(counts)
    for k in sums:
        assert (reach >> k) & 1
