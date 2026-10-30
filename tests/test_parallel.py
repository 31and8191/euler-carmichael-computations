"""Parallel prime generation and the gmpy2 backend must match the reference exactly."""

import numpy as np
import pytest

import pdc.primes as primes_module
from pdc.lam import Lambda
from pdc.primes import (
    generate_prime_set,
    _divisors,
    _proves_prime_py,
    _proves_prime_gmpy,
    HAVE_GMPY2,
)

# Above the serial threshold (200k divisors) so the process-pool path is exercised.
_PARALLEL_LAMBDA = Lambda.parse("2^12*3^7*5^4*7^3*11^2*13^2*17*19*23*29")


def test_parallel_matches_serial():
    """jobs>1 must yield the identical PrimeSet (same primes, same canonical order)."""
    serial = generate_prime_set(_PARALLEL_LAMBDA, jobs=1)
    parallel = generate_prime_set(_PARALLEL_LAMBDA, jobs=8)

    assert parallel.N == serial.N
    assert parallel.values == serial.values  # canonical (residue-lexsort) order is stable
    assert np.array_equal(parallel.residues, serial.residues)
    assert np.allclose(parallel.logs, serial.logs)


def test_strengthened_two_adic_pool_matches_serial(monkeypatch):
    """The experimental residue modulus must survive process-pool generation exactly."""
    lam = Lambda.parse("2^7*3^4*5^2*7^2*11*13*17*19*23")
    serial = generate_prime_set(
        lam, jobs=1, min_prime=72, two_adic_multiplier=2)

    # Exercise the process-pool branch on a bounded fixture rather than duplicating the
    # existing 50k-prime parallel test above.
    monkeypatch.setattr(primes_module, "_SERIAL_DIVISOR_THRESHOLD", 0)
    parallel = generate_prime_set(
        lam, jobs=2, min_prime=72, two_adic_multiplier=2)

    assert parallel.values == serial.values
    assert parallel.stage_moduli == serial.stage_moduli
    assert np.array_equal(parallel.residues, serial.residues)
    assert np.allclose(parallel.logs, serial.logs)


def test_four_q_pool_matches_serial_exactly(monkeypatch):
    lam = Lambda.parse("2^7*3^4*5^2*7^2*11*13*17*19*23")
    serial = generate_prime_set(
        lam, jobs=1, min_prime=72, two_adic_multiplier=4)

    monkeypatch.setattr(primes_module, "_SERIAL_DIVISOR_THRESHOLD", 0)
    parallel = generate_prime_set(
        lam, jobs=2, min_prime=72, two_adic_multiplier=4)

    assert parallel.values == serial.values
    assert parallel.stage_moduli == serial.stage_moduli
    assert np.array_equal(parallel.residues, serial.residues)
    assert np.array_equal(parallel.logs, serial.logs)


def test_generated_primes_valid_at_parallel_scale():
    """Every emitted p is a unit with p-1 | Lambda, and its residue row is correct."""
    P = generate_prime_set(_PARALLEL_LAMBDA, jobs=8)
    L = _PARALLEL_LAMBDA.value
    Qs = _PARALLEL_LAMBDA.prime_powers
    assert P.N > 0
    # Spot-check a sample (full check is O(N*r); the sample keeps the test fast).
    for j in range(0, P.N, 97):
        v = P.values[j]
        assert L % (v - 1) == 0
        assert L % v != 0
        for k, Q in enumerate(Qs):
            assert int(P.residues[j, k]) == v % Q
    # Primes are distinct.
    assert len(set(P.values)) == P.N


@pytest.mark.skipif(not HAVE_GMPY2, reason="gmpy2 not installed")
def test_gmpy2_backend_agrees_with_pure_python():
    """The two primality backends must return identical verdicts over a real sweep."""
    lam = Lambda.parse("2^5*3^3*5^2*7*11*13")
    mismatches = 0
    primes = 0
    for d, dprimes in _divisors(lam):
        n = d + 1
        g = _proves_prime_gmpy(n, dprimes)
        p = _proves_prime_py(n, dprimes)
        assert g == p, f"backend disagreement at {n}: gmpy2={g} pure={p}"
        primes += int(p)
    assert primes > 0
