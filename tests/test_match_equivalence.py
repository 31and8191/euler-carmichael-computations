"""The Cython matcher must produce *identical* pairings to the pure-Python reference."""

import numpy as np
import pytest

from pdc.lam import Lambda
from pdc.primes import generate_prime_set
from pdc import pushdown
from pdc.pushdown import _match_stage_py, push_down


def _partition(result):
    """The output as a canonical set of frozensets of prime values (order-independent)."""
    vals = result.primes.values
    return frozenset(frozenset(vals[j] for j in g) for g in result.base)


@pytest.mark.skipif(not pushdown.HAVE_CYTHON, reason="Cython extension not built")
def test_cython_kernel_matches_python_on_random_stages():
    rng = np.random.default_rng(12345)
    for _ in range(200):
        C = int(rng.integers(0, 40))
        r = int(rng.integers(1, 8))
        # small residue alphabet so prefixes actually collide
        cres = rng.integers(0, 4, size=(C, r)).astype(np.int64)
        cinv = rng.integers(0, 4, size=(C, r)).astype(np.int64)
        order = np.lexsort(cres.T[::-1]).astype(np.int64) if C else np.empty(0, np.int64)
        m = int(rng.integers(0, r))

        p_cy, c_cy = pushdown._cy_match.match_stage(
            np.ascontiguousarray(cres), np.ascontiguousarray(cinv),
            np.ascontiguousarray(order), m)
        p_py, c_py = _match_stage_py(cres, cinv, order, m)

        assert np.array_equal(p_cy, p_py)
        assert np.array_equal(np.asarray(c_cy), np.asarray(c_py))


@pytest.mark.skipif(not pushdown.HAVE_CYTHON, reason="Cython extension not built")
def test_full_run_identical_with_and_without_cython(monkeypatch):
    lam = Lambda.parse("2^6 * 3^3 * 5^2 * 7^2 * 11 * 13 * 17 * 19")
    P = generate_prime_set(lam)

    res_cy = push_down(P)

    monkeypatch.setattr(pushdown, "HAVE_CYTHON", False)
    res_py = push_down(P)

    assert sorted(res_cy.factor_counts) == sorted(res_py.factor_counts)
    assert _partition(res_cy) == _partition(res_py)


@pytest.mark.skipif(not pushdown.HAVE_CYTHON, reason="Cython extension not built")
def test_strengthened_full_run_identical_with_and_without_cython(monkeypatch):
    """The compiled matcher must also respect a strengthened residue column."""
    lam = Lambda.parse("2^7*3^4*5^2*7^2*11*13*17*19*23")
    P = generate_prime_set(lam, min_prime=72, two_adic_multiplier=4)

    res_cy = push_down(P)
    monkeypatch.setattr(pushdown, "HAVE_CYTHON", False)
    res_py = push_down(P)

    assert res_cy.stages == res_py.stages
    assert res_cy.factor_counts == res_py.factor_counts
    assert _partition(res_cy) == _partition(res_py)
