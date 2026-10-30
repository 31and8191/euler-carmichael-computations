"""Independent verification that constructed numbers really are Carmichael.

A composite ``n`` is a Carmichael number iff it is squarefree and ``p - 1 | n - 1`` for
every prime ``p | n`` (Korselt's criterion), and it has at least three prime factors.

Two checkers live here:

* :func:`is_carmichael` reconstructs the exact integer ``n`` from its factors and applies
  Korselt directly.  It is fully self-contained and is used to validate factored files
  (e.g. the original 1998 ``carmichaels.golden``), where the numbers are small.

* :func:`verify_base_numbers` validates the push-down's own output at scale.  There a
  single base can have tens of millions of prime factors, so forming ``n`` (hundreds of
  millions of digits) is infeasible.  Instead it uses the **Korselt-via-Lambda** argument,
  which needs no giant integer:

      every ``p in P`` has ``(p-1) | Lambda`` and the push-down forces ``n == 1`` modulo
      every stage modulus, hence ``n == 1 (mod Lambda)``,
      hence ``(p-1) | Lambda | (n-1)`` for every factor ``p`` -- Korselt's divisibility,
      for free.

  So it suffices to check, independently of the push-down's internal state: (i) every
  prime really does satisfy ``(p-1) | Lambda``; (ii) each base is squarefree (distinct
  factors) with ``>= 3`` of them; and (iii) ``n == 1 (mod Lambda)``, recomputed from the
  primes' residues as a per-column modular product -- an ``O(K)`` small-integer reduction
  rather than an ``O(K^2)`` big-integer product.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pushdown import _modprod_rows


@dataclass
class CarmichaelCheck:
    n: int
    factors: list[int]
    ok: bool
    reason: str = ""


def is_carmichael(factors: list[int]) -> CarmichaelCheck:
    """Verify Korselt's criterion for ``n = prod(factors)``."""
    if len(factors) < 3:
        n = 1
        for p in factors:
            n *= p
        return CarmichaelCheck(n, factors, False, f"only {len(factors)} prime factor(s)")
    if len(set(factors)) != len(factors):
        return CarmichaelCheck(0, factors, False, "not squarefree (repeated prime)")
    n = 1
    for p in factors:
        n *= p
    nm1 = n - 1
    for p in factors:
        if nm1 % (p - 1) != 0:
            return CarmichaelCheck(n, factors, False, f"(p-1) does not divide n-1 for p={p}")
    return CarmichaelCheck(n, factors, True)


def primes_divide_lambda(values, lam_value) -> int | None:
    """Independent recheck of the defining property of ``P``: every prime ``p`` has
    ``(p-1) | Lambda``.  Returns the first offending prime, or ``None`` if all pass."""
    return next((p for p in values if lam_value % (p - 1) != 0), None)


def _n_mod_stage_moduli_is_one(idx: np.ndarray, residues, Qrow_pd, prod_dtype, r,
                               chunk) -> bool:
    """``n == 1`` modulo every push-down stage modulus for the factors ``idx``.

    This is a chunked ``O(K)`` small-integer reduction with no giant ``n`` formed.  In
    the experimental class-A path the 2-primary stage may be stronger than the
    corresponding factor of ``Lambda``.
    """
    acc = np.ones(r, dtype=prod_dtype)
    for s in range(0, idx.size, chunk):
        part = _modprod_rows(residues[idx[s:s + chunk]], Qrow_pd, prod_dtype)
        acc = (acc * part.astype(prod_dtype)) % Qrow_pd
    return bool(np.all(acc.astype(np.int64) == 1))


def _check_bases(base_chunk, residues, values, Qrow_pd, prod_dtype, r, chunk
                 ) -> tuple[list[CarmichaelCheck], list[CarmichaelCheck]]:
    """Verify a list of base numbers (each a list of prime indices) via Korselt-via-Lambda.
    Returns ``(carmichael, other)`` -- the per-base worker body shared by the serial and
    parallel paths."""
    carmichael: list[CarmichaelCheck] = []
    other: list[CarmichaelCheck] = []
    for indices in base_chunk:
        K = len(indices)
        if K < 3:
            other.append(CarmichaelCheck(0, [], False, f"only {K} prime factor(s)"))
            continue
        idx = np.asarray(indices, dtype=np.int64)
        if int(np.unique(idx).size) != K:
            carmichael.append(CarmichaelCheck(
                0, [values[j] for j in indices], False, "not squarefree (repeated prime)"))
        elif not _n_mod_stage_moduli_is_one(
                idx, residues, Qrow_pd, prod_dtype, r, chunk):
            carmichael.append(CarmichaelCheck(
                0, [values[j] for j in indices], False,
                "n != 1 modulo every required push-down modulus"))
        else:
            # squarefree, >= 3 factors, and n == 1 (mod Lambda) with every (p-1) | Lambda
            # => (p-1) | (n-1) for all p: Korselt holds.  (factors omitted to save memory.)
            carmichael.append(CarmichaelCheck(0, [], True))
    return carmichael, other


# Fork-inherited state for the parallel base-number verifier (avoids pickling the big
# residues/values arrays into each worker).  Mirrors ``euler._ROW_G``.
_VBN_G: dict = {}


def _vbn_worker(base_chunk):
    g = _VBN_G
    return _check_bases(base_chunk, g["residues"], g["values"],
                        g["Qrow_pd"], g["prod_dtype"], g["r"], g["chunk"])


def verify_base_numbers(result, *, chunk: int = 1 << 20, jobs: int | None = 1
                        ) -> tuple[list[CarmichaelCheck], list[CarmichaelCheck]]:
    """Verify every base number of a :class:`~pdc.pushdown.PushdownResult` at scale.

    Returns ``(carmichael, other)``: those with ``>= 3`` factors (each flagged ``ok`` iff
    it passes), and the 1- or 2-factor building blocks.  Uses the Korselt-via-Lambda
    argument (see the module docstring) -- no base integer is ever formed.

    Each base number is independent, so with ``jobs > 1`` (or ``jobs=None`` -> auto) the
    base list is sharded across worker processes.  Workers fork-inherit the shared
    ``residues``/``values`` arrays (no pickling of the big pool), mirroring
    :func:`pdc.euler._hard_rows`.  Results are returned in input order.
    """
    lam = result.lam
    lam_value = lam.value
    Qs = result.primes.stage_moduli
    r = len(Qs)
    residues = result.primes.residues
    values = result.primes.values

    # (i) every prime satisfies (p-1) | Lambda  (checked once over all of P).
    bad_prime = primes_divide_lambda(values, lam_value)
    assert bad_prime is None, f"prime {bad_prime}: (p-1) does not divide Lambda"

    Qrow = np.asarray(Qs, dtype=np.int64)
    prod_dtype = np.int64 if (int(Qrow.max()) - 1) ** 2 < (1 << 62) else object
    Qrow_pd = Qrow.astype(object) if prod_dtype is object else Qrow

    if jobs is None:
        import os
        jobs = min(os.cpu_count() or 1, 72)

    # Serial for small inputs or when unparallelized (keeps tests deterministic & simple).
    if jobs <= 1 or len(result.base) < 256:
        return _check_bases(result.base, residues, values, Qrow_pd, prod_dtype, r, chunk)

    from concurrent.futures import ProcessPoolExecutor
    _VBN_G.update(residues=residues, values=values, Qrow_pd=Qrow_pd,
                  prod_dtype=prod_dtype, r=r, chunk=chunk)
    n = len(result.base)
    step = max(1, n // (8 * jobs))
    chunks = [result.base[i:i + step] for i in range(0, n, step)]
    carmichael: list[CarmichaelCheck] = []
    other: list[CarmichaelCheck] = []
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for carm_part, other_part in ex.map(_vbn_worker, chunks):
            carmichael.extend(carm_part)
            other.extend(other_part)
    return carmichael, other


def read_factored_carmichaels(path: str) -> list[list[int]]:
    """Read a file of Carmichael numbers in factored form (one prime per line, blank
    line between numbers) -- the format of ``carmichaels.golden`` / ``carmichaels``."""
    groups: list[list[int]] = []
    current: list[int] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                if current:
                    groups.append(current)
                    current = []
                continue
            current.append(int(line.split()[0]))
    if current:
        groups.append(current)
    return groups
