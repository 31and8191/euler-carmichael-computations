"""A modern re-implementation of Algorithm 1 ("Many Carmichaels subset-product") from

    W. R. Alford, J. Grantham, S. Hayman, A. Shallue,
    "Constructing Carmichael numbers through improved subset-product algorithms",
    Math. Comp. 83 (2014).

The original artifact was the C program ``pushdown.c``; this package is a clean,
self-contained Python port that

  * generates the prime set ``P`` from a chosen modulus ``Lambda`` (with rigorous
    primality proofs, since ``p - 1`` is fully factored),
  * runs the push-down subset-product algorithm to produce coprime "base" Carmichaels,
  * verifies every output via Korselt's criterion, and
  * reports which factor counts ``k`` are achievable.
"""

from .lam import Lambda, DEFAULT_LAMBDA
from .primes import PrimeSet, generate_prime_set, proves_prime
from .pushdown import push_down, PushdownResult
from .verify import is_carmichael, verify_base_numbers, read_factored_carmichaels
from .coverage import reachable_set, reachable_counts, achievable_ks, format_ranges, base_histogram
from .euler import (
    construct_euler_pseudoprime,
    construct_maximal,
    verify_construction,
    structural_check,
    f2_kernel_vector,
    first_primes,
    EulerResult,
    EulerCheck,
    StructuralCheck,
)

__all__ = [
    "Lambda",
    "DEFAULT_LAMBDA",
    "PrimeSet",
    "generate_prime_set",
    "proves_prime",
    "push_down",
    "PushdownResult",
    "is_carmichael",
    "verify_base_numbers",
    "read_factored_carmichaels",
    "reachable_set",
    "reachable_counts",
    "achievable_ks",
    "format_ranges",
    "base_histogram",
    "construct_euler_pseudoprime",
    "construct_maximal",
    "verify_construction",
    "structural_check",
    "StructuralCheck",
    "f2_kernel_vector",
    "first_primes",
    "EulerResult",
    "EulerCheck",
]
