"""The modulus ``Lambda`` of the Erdos construction.

A Carmichael number can be built from a set of primes ``p`` with ``p - 1 | Lambda``
(and ``p`` not dividing ``Lambda``): any subset whose product is ``== 1 (mod Lambda)``
is squarefree and satisfies Korselt's criterion, hence is Carmichael.

This module fixes the notation used throughout the package, following the paper
"Constructing Carmichael numbers through improved subset-product algorithms"
(Alford, Grantham, Hayman, Shallue, Math. Comp. 2014):

    Lambda = prod_i q_i ** h_i          and          Q_i = q_i ** h_i

The original C implementation (``pushdown.c``) stored each prime as its vector of
residues modulo the prime powers ``Q_i``, ordered by *descending value* of ``Q_i``.
We keep that ordering: it is what makes the greedy "push down" step (see
``pushdown.py``) clear the largest prime powers first, which is the most valuable
progress toward the identity in ``(Z/Lambda Z)^*``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Lambda:
    """A fully factored modulus ``Lambda``.

    Attributes
    ----------
    factors:
        The pairs ``(q_i, h_i)`` in the *canonical column order* used by the
        algorithm: descending value of ``Q_i = q_i ** h_i``.
    """

    factors: tuple[tuple[int, int], ...]

    # -- constructors ----------------------------------------------------

    @classmethod
    def from_pairs(cls, pairs) -> "Lambda":
        """Build from an iterable of ``(prime, exponent)`` pairs.

        Pairs are validated (distinct primes, positive exponents) and reordered
        into the canonical descending-``Q_i`` column order.
        """
        seen: set[int] = set()
        clean: list[tuple[int, int]] = []
        for q, h in pairs:
            q, h = int(q), int(h)
            if h <= 0:
                raise ValueError(f"exponent for {q} must be >= 1, got {h}")
            if q in seen:
                raise ValueError(f"prime {q} repeated in Lambda specification")
            if not _is_small_prime(q):
                raise ValueError(f"base {q} is not prime")
            seen.add(q)
            clean.append((q, h))
        # Canonical order: largest prime power first.
        clean.sort(key=lambda qh: qh[0] ** qh[1], reverse=True)
        return cls(tuple(clean))

    @classmethod
    def parse(cls, text: str) -> "Lambda":
        """Parse a human spec like ``"2^10 * 3^6 * 5^4 * 7^2 * 11"``.

        ``*`` separates factors; ``^`` denotes exponentiation (default exponent 1).
        Whitespace is ignored.
        """
        pairs = []
        for term in text.replace(" ", "").split("*"):
            if not term:
                continue
            m = re.fullmatch(r"(\d+)(?:\^(\d+))?", term)
            if not m:
                raise ValueError(f"cannot parse factor {term!r} in Lambda spec")
            q = int(m.group(1))
            h = int(m.group(2)) if m.group(2) else 1
            pairs.append((q, h))
        if not pairs:
            raise ValueError("empty Lambda specification")
        return cls.from_pairs(pairs)

    # -- derived quantities ---------------------------------------------

    @property
    def r(self) -> int:
        """Number of distinct prime powers (= number of residue columns / stages)."""
        return len(self.factors)

    @property
    def primes(self) -> tuple[int, ...]:
        """The base primes ``q_i`` in canonical column order."""
        return tuple(q for q, _ in self.factors)

    @property
    def prime_powers(self) -> tuple[int, ...]:
        """The prime powers ``Q_i = q_i ** h_i`` in canonical column order."""
        return tuple(q ** h for q, h in self.factors)

    @property
    def value(self) -> int:
        """The integer ``Lambda`` itself."""
        v = 1
        for Q in self.prime_powers:
            v *= Q
        return v

    @property
    def prime_set(self) -> frozenset[int]:
        """The set ``{q_i}`` of primes dividing ``Lambda`` (used to enforce ``p ∤ Lambda``)."""
        return frozenset(self.primes)

    def k_estimate(self) -> int:
        """The Loh--Niebuhr estimate ``K(Lambda)`` of ``|P|`` (Section 2 of the paper).

        ``K(Lambda) = floor( Lambda / (phi(Lambda) * ln sqrt(2 Lambda))
                              * prod_i (h_i + (q_i - 2)/(q_i - 1)) )``
        """
        lam = self.value
        phi = 1
        for q, h in self.factors:
            phi *= q ** (h - 1) * (q - 1)
        prod = 1.0
        for q, h in self.factors:
            prod *= h + (q - 2) / (q - 1)
        return int(lam / (phi * math.log(math.sqrt(2 * lam))) * prod)

    def __str__(self) -> str:
        return " * ".join(f"{q}^{h}" if h > 1 else f"{q}" for q, h in self.factors)


def _is_small_prime(n: int) -> bool:
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


# A modest default Lambda: a few thousand divisors, generating a prime set P of a
# few hundred primes in a second or two, with enough spread of factor counts to
# cover a wide band of k.  Override with ``--lambda`` on the CLI for larger runs.
DEFAULT_LAMBDA = Lambda.parse("2^7 * 3^4 * 5^2 * 7^2 * 11 * 13 * 17 * 19 * 23")


# Named presets selectable with ``--preset`` on the CLI.  ``record-alg1`` has
# more than 19,565,220 candidate primes, but its base-number factor counts do
# not give contiguous coverage through that endpoint.
PRESETS = {
    # K(Lambda) ~ 24.5M ; ~230M divisors ; r = 19 ; 128-bit moduli.
    "record-alg1": "2^12*3^7*5^4*7^3*11^2*13^2*17^2*19*23*29*31*37*41*43*47*53*59*61*67",
    # "Flat" variant: many distinct primes, low exponents -> the push-down emits many
    # base numbers whose subset sums cover a long contiguous interval.
    # K ~ 21.5M ; ~212M divisors ; r = 21.
    "record-flat": "2^9*3^5*5^3*7^2*11^2*13^2*17*19*23*29*31*37*41*43*47*53*59*61*67*71*73",
    # 1998 example: K(Lambda) ~ 1.35M ; ~10.6M divisors ; r = 16.
    "lambda-1998": "2^10*3^6*5^4*7^2*11^2*13^2*17*19*23*29*31*37*41*43*47*53",
    # The built-in small demo (a couple thousand divisors).
    "demo": "2^7*3^4*5^2*7^2*11*13*17*19*23",
}


def preset(name: str) -> "Lambda":
    """Return the named preset Lambda (see :data:`PRESETS`)."""
    try:
        return Lambda.parse(PRESETS[name])
    except KeyError:
        raise ValueError(f"unknown preset {name!r}; choices: {', '.join(PRESETS)}") from None
