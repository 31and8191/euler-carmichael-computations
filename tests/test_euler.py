"""The Euler-pseudoprime construction (pdc.euler) must produce genuine Euler pseudoprimes.

The strong test is the last one: build the number by F2 linear algebra, form the integer
``n`` exactly, and run the *literal* Solovay-Strassen test ``a**((n-1)/2) == (a/n) (mod n)``
with gmpy2 -- a check that knows nothing about class A or our construction.
"""

import random

import gmpy2
import numpy as np

from pdc.lam import Lambda
from pdc.primes import generate_prime_set, _v2
from pdc.pushdown import push_down
from pdc.euler import (
    construct_euler_pseudoprime,
    construct_maximal,
    verify_construction,
    structural_check,
    f2_kernel_vector,
    first_primes,
    _legendre_bits,
    _rank_f2,
)


def test_first_primes():
    assert first_primes(0) == []
    assert first_primes(1) == [2]
    assert first_primes(10) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    assert first_primes(100)[-1] == 541  # 100th prime


def test_legendre_bits_match_gmpy2():
    """The vectorized reciprocity path must equal gmpy2's Jacobi symbol prime-by-prime."""
    rng = random.Random(1234)
    # a spread of primes larger than the bases we test
    primes = [p for p in range(101, 5000) if gmpy2.is_prime(p)]
    vals = [primes[rng.randrange(len(primes))] for _ in range(400)]
    vals_obj = np.array(vals, dtype=object)
    p_mod8 = (vals_obj % 8).astype(np.int8)
    mask64 = (1 << 64) - 1
    limbs = [((vals_obj >> (64 * i)) & mask64).astype(np.uint64) for i in range(1)]
    for base in [2, 3, 5, 7, 11, 13, 97, 101]:
        bits = _legendre_bits(base, limbs, p_mod8)
        for p, b in zip(vals, bits.tolist()):
            expected = 1 if gmpy2.jacobi(base, p) < 0 else 0
            assert b == expected, (base, p, b, expected)


def test_f2_kernel_random():
    """A returned kernel vector must be nonzero and annihilate every row over F2."""
    rng = random.Random(7)
    for _ in range(200):
        ncols = rng.randint(2, 40)
        nrows = rng.randint(1, 30)
        rows = [rng.getrandbits(ncols) for _ in range(nrows)]
        sigma = f2_kernel_vector(rows, ncols)
        rank = _rank_f2(rows)
        if rank >= ncols:
            assert sigma is None
            continue
        assert sigma is not None and sigma != 0
        assert sigma < (1 << ncols)
        for r in rows:
            assert bin(r & sigma).count("1") % 2 == 0  # <r, sigma> == 0 over F2


def test_f2_kernel_sparse_is_lower_weight():
    rng = random.Random(99)
    rows = [rng.getrandbits(60) for _ in range(20)]
    plain = f2_kernel_vector(rows, 60)
    sparse = f2_kernel_vector(rows, 60, want_sparse=True)
    assert sparse is not None and plain is not None
    assert bin(sparse).count("1") <= bin(plain).count("1")
    for r in rows:  # still a valid kernel vector
        assert bin(r & sparse).count("1") % 2 == 0


def _build(spec: str, t: int, want_sparse=False):
    lam = Lambda.parse(spec)
    maxbase = first_primes(t)[-1]
    P = generate_prime_set(lam, min_prime=maxbase + 1, class_a=True)
    res = push_down(P)
    return lam, P, construct_euler_pseudoprime(res, t, want_sparse=want_sparse)


def test_construction_is_class_A_and_coprime():
    lam, P, eres = _build("2^7*3^4*5^2*7^2*11*13*17*19*23", 20)
    assert eres.solved
    v2L = _v2(lam.value)
    # every factor: p-1 has 2-adic valuation strictly below v_2(Lambda) (=> product class A)
    for p in eres.factor_values:
        assert _v2(p - 1) < v2L
        assert p > eres.target_bases[-1]  # coprime to every test base
    assert len(set(eres.factor_values)) == len(eres.factor_values)  # squarefree


def test_construction_passes_literal_solovay_strassen_demo():
    """Gold standard: form n and run the actual Solovay-Strassen test for every base."""
    lam, P, eres = _build("2^7*3^4*5^2*7^2*11*13*17*19*23", 20)
    assert eres.solved
    n = eres.form_n()
    assert not gmpy2.is_prime(n)  # it must be composite
    e = (n - 1) // 2
    for a in eres.target_bases:
        lhs = gmpy2.powmod(a, e, n)
        rhs = gmpy2.jacobi(a, n) % n  # +1 -> 1 ; -1 -> n-1
        assert lhs == rhs, f"n is NOT an Euler pseudoprime to base {a}"
    # and the module's own verifier agrees (it also runs the modexp at this size)
    chk = verify_construction(eres)
    assert chk.ok and chk.class_a and chk.failing_base is None
    assert chk.survived_bases == len(eres.target_bases)


def test_base_2_becomes_a_matrix_row_when_not_forced_mod_8():
    """The experimental mod-4 path must genuinely constrain base 2."""
    lam = Lambda.parse("2*3^4*5^2*7^2*11*13*17*19*23*29*31*37*41")
    P = generate_prime_set(lam, min_prime=30, two_adic_multiplier=2)
    res = push_down(P)
    eres = construct_euler_pseudoprime(res, 10)
    assert 2 in eres.hard_bases
    assert 2 not in eres.auto_bases
    assert eres.rank == 1                 # the base-2 row is nonzero, not merely listed
    assert eres.solved
    assert structural_check(eres).class_a


def test_shallue_style_full_pool_produces_class_a_euler_pseudoprime():
    """Keep all p-1|Lambda primes; double only the 2-primary push-down modulus."""
    spec = "2^7*3^4*5^2*7^2*11*13*17*19*23"
    lam = Lambda.parse(spec)
    maxbase = first_primes(20)[-1]
    filtered = generate_prime_set(lam, min_prime=maxbase + 1, class_a=True)
    full = generate_prime_set(lam, min_prime=maxbase + 1, two_adic_multiplier=2)
    assert full.N > filtered.N

    res = push_down(full)
    two_col = lam.primes.index(2)
    assert res.stages[two_col].Q == 2 * lam.prime_powers[two_col]

    eres = construct_euler_pseudoprime(res, 20)
    assert eres.solved
    chk = verify_construction(eres)
    assert chk.ok and chk.class_a and chk.failing_base is None


def test_four_q_full_pool_uses_exact_stage_and_produces_class_a():
    spec = "2^7*3^4*5^2*7^2*11*13*17*19*23"
    lam = Lambda.parse(spec)
    maxbase = first_primes(20)[-1]
    two_q = generate_prime_set(lam, min_prime=maxbase + 1, two_adic_multiplier=2)
    four_q = generate_prime_set(lam, min_prime=maxbase + 1, two_adic_multiplier=4)
    assert set(four_q.values) == set(two_q.values)

    res = push_down(four_q)
    two_col = lam.primes.index(2)
    assert res.stages[two_col].Q == 4 * lam.prime_powers[two_col]
    assert res.stages[two_col].Q != 2 * lam.prime_powers[two_col]

    eres = construct_maximal(res, 20)
    assert eres.solved
    assert eres.target_bases
    chk = verify_construction(eres, gold_modexp_bit_limit=0)
    assert chk.ok and chk.class_a and chk.failing_base is None


def test_construction_scales_to_more_bases():
    """A flat Lambda should let us beat the '211 paper' 47-base Euler pseudoprime.

    Here n is large (~10^4 digits), so we use the rigorous class-A + Jacobi verification
    (independent of the kernel) rather than the slow literal modexp; the demo test above
    exercises the gold-standard Solovay-Strassen modexp on a small n.
    """
    lam, P, eres = _build("2^8*3^5*5^3*7^2*11^2*13^2*17*19*23*29*31*37", 60, want_sparse=True)
    assert eres.solved, "expected a kernel solution for the first 60 primes"
    chk = verify_construction(eres, gold_modexp_bit_limit=0)  # skip the minute-long modexp
    assert chk.class_a and chk.n_mod_lambda_is_one and chk.squarefree
    assert chk.failing_base is None, f"failed at base {chk.failing_base}"
    assert chk.survived_bases == 60 > 47
