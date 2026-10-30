"""Primality proofs and prime-set generation."""

from pdc.lam import Lambda
from pdc.primes import proves_prime, generate_prime_set


def test_proves_prime_small():
    # n, prime divisors of n-1
    assert proves_prime(7, {2, 3})        # 6 = 2*3
    assert proves_prime(13, {2, 3})       # 12 = 2^2*3
    assert proves_prime(2, set())
    assert not proves_prime(1, set())
    assert not proves_prime(9, {2})       # 8 = 2^3 ; 9 composite
    assert not proves_prime(15, {2, 7})   # 14 = 2*7 ; 15 composite
    assert not proves_prime(561, {2, 5, 7})  # Carmichael, composite


def test_proves_prime_rejects_pseudoprime():
    # 341 = 11*31 is a base-2 Fermat pseudoprime; the Lucas proof must reject it.
    # 340 = 2^2 * 5 * 17
    assert not proves_prime(341, {2, 5, 17})


def test_generated_primes_satisfy_construction():
    lam = Lambda.parse("2^5 * 3^3 * 5^2 * 7 * 11 * 13")
    P = generate_prime_set(lam)
    L = lam.value
    assert P.N > 0
    for v, row in zip(P.values, P.residues):
        # p is prime, p-1 | Lambda, p does not divide Lambda
        assert L % (v - 1) == 0
        assert L % v != 0
        # residue vector matches
        for k, Q in enumerate(lam.prime_powers):
            assert int(row[k]) == v % Q


def test_strengthened_two_adic_stage_keeps_full_pool():
    lam = Lambda.parse("2^4 * 3^3 * 5^2 * 7 * 11 * 13")
    ordinary = generate_prime_set(lam, jobs=1)
    strengthened = generate_prime_set(lam, jobs=1, two_adic_multiplier=2)

    # The admissible-prime definition is unchanged; only its residue representation differs.
    assert set(strengthened.values) == set(ordinary.values)
    two_col = lam.primes.index(2)
    assert strengthened.stage_moduli[two_col] == 2 * lam.prime_powers[two_col]
    for k, Q in enumerate(strengthened.stage_moduli):
        assert all(int(row[k]) == p % Q
                   for p, row in zip(strengthened.values, strengthened.residues))


def test_four_q_stage_keeps_same_pool_and_is_distinct_from_two_q():
    lam = Lambda.parse("2^4 * 3^3 * 5^2 * 7 * 11 * 13")
    ordinary = generate_prime_set(lam, jobs=1)
    two_q = generate_prime_set(lam, jobs=1, two_adic_multiplier=2)
    four_q = generate_prime_set(lam, jobs=1, two_adic_multiplier=4)

    assert set(four_q.values) == set(ordinary.values) == set(two_q.values)
    two_col = lam.primes.index(2)
    Q = lam.prime_powers[two_col]
    assert two_q.stage_moduli[two_col] == 2 * Q
    assert four_q.stage_moduli[two_col] == 4 * Q
    assert two_q.stage_moduli != four_q.stage_moduli
    for k, modulus in enumerate(four_q.stage_moduli):
        assert all(int(row[k]) == p % modulus
                   for p, row in zip(four_q.values, four_q.residues))


def test_class_a_strategies_are_explicit_alternatives():
    import pytest

    lam = Lambda.parse("2^4 * 3^3 * 5^2 * 7 * 11 * 13")
    with pytest.raises(ValueError, match="either the class_a prime filter"):
        generate_prime_set(lam, jobs=1, class_a=True, two_adic_multiplier=2)
    with pytest.raises(ValueError, match="positive power of two"):
        generate_prime_set(lam, jobs=1, two_adic_multiplier=3)
    for invalid in (0, -2, 6):
        with pytest.raises(ValueError, match="positive power of two"):
            generate_prime_set(lam, jobs=1, two_adic_multiplier=invalid)
