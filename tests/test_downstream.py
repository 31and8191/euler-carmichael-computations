"""The scalable push-down + Lambda-verifier must agree with exact reconstruction."""

from pdc.lam import Lambda
from pdc.primes import generate_prime_set
from pdc.pushdown import push_down
from pdc.verify import verify_base_numbers, is_carmichael, primes_divide_lambda

# A Lambda large enough to trigger the discard ("throw away one number") branch and
# produce a spread of base sizes, but small enough to reconstruct exactly for comparison.
_SPEC = "2^9*3^5*5^3*7^2*11^2*13*17*19*23"


def test_lambda_verifier_agrees_with_exact_reconstruction():
    lam = Lambda.parse(_SPEC)
    P = generate_prime_set(lam, jobs=4)
    result = push_down(P)

    carm, other = verify_base_numbers(result)
    assert all(c.ok for c in carm), "Lambda-verifier flagged a base it should not have"

    # Exact, independent ground truth per base.
    exact_ok = 0
    for indices in result.base:
        factors = [P.values[j] for j in indices]
        chk = is_carmichael(factors)
        if len(factors) >= 3:
            assert chk.ok, f"exact check rejects a base the algorithm produced: {chk.reason}"
            exact_ok += 1
    # >=3-factor bases land in `carm`; 1- or 2-factor building blocks land in `other`.
    assert len(carm) == exact_ok
    assert len(other) == sum(1 for g in result.base if len(g) < 3)


def test_forest_flatten_is_a_disjoint_cover():
    """Every base is squarefree (distinct indices) and bases are pairwise disjoint."""
    lam = Lambda.parse(_SPEC)
    P = generate_prime_set(lam, jobs=4)
    result = push_down(P)

    seen: set[int] = set()
    for indices in result.base:
        assert len(set(indices)) == len(indices)        # distinct within a base
        assert seen.isdisjoint(indices)                 # disjoint across bases
        seen.update(indices)
    # factor_counts is just the per-base size.
    assert result.factor_counts == [len(g) for g in result.base]


def test_primes_divide_lambda_holds():
    lam = Lambda.parse(_SPEC)
    P = generate_prime_set(lam, jobs=4)
    assert primes_divide_lambda(P.values, lam.value) is None
