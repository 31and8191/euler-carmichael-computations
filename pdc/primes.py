"""Step 2 of the Erdos construction: build the prime set ``P``.

``P = { p prime : p - 1 | Lambda  and  p ∤ Lambda }``.

We enumerate every divisor ``d`` of ``Lambda`` and test ``p = d + 1`` for primality.
Crucially, ``p - 1 = d`` divides ``Lambda`` and so is *completely factored* (its prime
divisors are a subset of the ``q_i``).  That lets us give a genuine **primality proof**
via the converse of Fermat's little theorem (Lucas / Pocklington with fully factored
``n - 1``) rather than relying on a probabilistic test -- the paper notes this matters,
since pseudoprimes do turn up here.

The result is packaged as a :class:`PrimeSet`: the exact prime values, their residue
vectors modulo the ``Q_i`` (the representation the push-down algorithm operates on),
and natural logs (used only as a size tie-breaker / for reporting digit counts).
"""

from __future__ import annotations

import itertools
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from .lam import Lambda

# Optional gmpy2 acceleration for the modular exponentiation in the primality proof.
# When present, ``gmpy2.powmod`` (GMP-backed) is roughly an order of magnitude faster
# than CPython's ``pow`` on the ~100--130-bit moduli that arise here.  The package runs
# unchanged without it (pure-Python fallback below), mirroring the ``HAVE_CYTHON`` switch
# in :mod:`pdc.pushdown`.
try:  # pragma: no cover - exercised only when the optional dep is installed
    from gmpy2 import mpz as _mpz, powmod as _powmod, is_strong_prp as _is_strong_prp

    HAVE_GMPY2 = True
except ImportError:  # pragma: no cover
    HAVE_GMPY2 = False

# Small primes used to (a) reject n sharing a small factor and (b) draw Lucas witnesses
# from.  A primitive root of a prime n is found among the first few of these almost
# always, so the witness search below terminates immediately in practice.
_SMALL_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def _proves_prime_py(n: int, prime_divisors_of_n_minus_1) -> bool:
    """Pure-Python rigorous primality proof (fallback when gmpy2 is absent)."""
    if n < 2:
        return False
    if n < 4:
        return True  # 2, 3
    if n % 2 == 0:
        return False
    nm1 = n - 1
    # Cheap base-2 strong-probable-prime pre-filter: rejects the vast majority of
    # composite candidates before the (more expensive) witness search.
    if not _sprp2_py(n, nm1):
        return False
    # Lucas's converse of Fermat: n is prime iff some witness a has a^(n-1) == 1 (mod n)
    # and a^((n-1)/q) != 1 (mod n) for every prime q | n-1.  n-1's prime divisors are
    # supplied by the caller (a subset of Lambda's primes), so n-1 is fully factored.
    cofactors = [nm1 // q for q in set(prime_divisors_of_n_minus_1)]
    for a in _SMALL_PRIMES:
        if a >= n:
            break
        if pow(a, nm1, n) != 1:
            return False  # definitive: composite (Fermat witness)
        if all(pow(a, c, n) != 1 for c in cofactors):
            return True  # definitive: prime (primitive-root witness)
    # No witness among the small primes -> fall back to a wider scan (effectively never
    # reached for true primes, where a in {2,3,5} succeeds almost always).
    a = 41
    while a < 1 << 20:
        if pow(a, nm1, n) != 1:
            return False
        if all(pow(a, c, n) != 1 for c in cofactors):
            return True
        a += 1
    raise RuntimeError(f"primality proof inconclusive for {n}")


def _sprp2_py(n: int, nm1: int) -> bool:
    """Base-2 strong-probable-prime test (pure Python)."""
    d = nm1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    x = pow(2, d, n)
    if x == 1 or x == nm1:
        return True
    for _ in range(s - 1):
        x = x * x % n
        if x == nm1:
            return True
    return False


def _proves_prime_gmpy(n: int, prime_divisors_of_n_minus_1) -> bool:
    """gmpy2-accelerated rigorous primality proof (identical verdict to the pure path)."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    m = _mpz(n)
    nm1 = m - 1
    if not _is_strong_prp(m, 2):  # base-2 strong-probable-prime pre-filter
        return False
    cofactors = [nm1 // q for q in set(prime_divisors_of_n_minus_1)]
    one = _mpz(1)
    for a in _SMALL_PRIMES:
        if a >= n:
            break
        am = _mpz(a)
        if _powmod(am, nm1, m) != one:
            return False
        if all(_powmod(am, c, m) != one for c in cofactors):
            return True
    a = 41
    while a < 1 << 20:
        am = _mpz(a)
        if _powmod(am, nm1, m) != one:
            return False
        if all(_powmod(am, c, m) != one for c in cofactors):
            return True
        a += 1
    raise RuntimeError(f"primality proof inconclusive for {n}")


# Bind the active backend once (one attribute lookup saved per call in the hot loop).
proves_prime = _proves_prime_gmpy if HAVE_GMPY2 else _proves_prime_py
proves_prime.__doc__ = (
    "Return ``True`` iff ``n`` is **proven** prime, given the prime divisors of ``n-1`` "
    "(a subset of Lambda's primes, so ``n-1`` is fully factored).  Uses a base-2 "
    "strong-PRP pre-filter followed by a rigorous Lucas/Pocklington witness search.  "
    "Dispatches to a gmpy2-accelerated implementation when available."
)


def _enumerate_subtree(factors, start, d0, used0):
    """Yield ``(d, prime_divisors_of_d)`` for every divisor ``d`` of the suffix
    ``prod_{i>=start} q_i^{e_i}`` scaled by the fixed prefix product ``d0``.

    ``used0`` is the tuple of primes already present in ``d0``.  With ``start=0``,
    ``d0=1``, ``used0=()`` this enumerates every divisor of ``Lambda`` -- the prefix
    arguments exist so a worker can own one branch of the recursion (one choice of the
    leading exponents) and enumerate only its own subtree.
    """
    primes = [q for q, _ in factors]
    exps = [h for _, h in factors]
    r = len(factors)

    def rec(i: int, d: int, used: tuple[int, ...]):
        if i == r:
            yield d, used
            return
        q = primes[i]
        # exponent 0 first (q not used), then 1..h_i (q used)
        yield from rec(i + 1, d, used)
        power = 1
        for e in range(1, exps[i] + 1):
            power *= q
            yield from rec(i + 1, d * power, used + (q,))

    yield from rec(start, d0, used0)


def _divisors(lam: Lambda):
    """Yield ``(d, prime_divisors_of_d)`` over every divisor ``d`` of ``Lambda``."""
    yield from _enumerate_subtree(list(lam.factors), 0, 1, ())


def _v2(d: int) -> int:
    """2-adic valuation of a positive integer ``d`` (``v_2(d)``)."""
    return (d & -d).bit_length() - 1


def _sweep(subtree, excluded, Qs, min_prime: int = 1, v2_cap: int | None = None):
    """Prove ``p = d + 1`` over a stream of divisors; return compact shard arrays.

    Returns ``(values, residues, logs)`` where ``values`` is a list of the proven
    primes, ``residues`` is an ``(n, r)`` ``int64`` array of ``p mod Q_k``, and ``logs``
    is an ``(n,)`` ``float64`` array of ``ln p``.  Shared by the serial path and the
    parallel workers so both run identical per-candidate logic.

    Two optional filters (used by the Euler-pseudoprime construction, see :mod:`pdc.euler`):

    * ``min_prime`` -- keep only primes ``p >= min_prime``.  Setting this above the
      largest test base guarantees every base Carmichael (and every subset product) is
      coprime to all test bases.
    * ``v2_cap`` -- reject primes with ``v_2(p - 1) >= v2_cap``.  With ``v2_cap = v_2(Lambda)``
      this drops the primes of maximal 2-adic valuation, forcing ``max_p v_2(p-1) < v_2(Lambda)
      <= v_2(n-1)`` for every subset product ``n`` -- i.e. every product lands in **class A**
      (even index), where an Euler pseudoprime is exactly a number with Jacobi symbol ``+1``.
    """
    prove = proves_prime  # bind once (gmpy2 or pure-Python backend)
    values: list[int] = []
    rows: list[list[int]] = []
    logs: list[float] = []
    for d, dprimes in subtree:
        p = d + 1
        if p in excluded:  # p | Lambda  -> reject
            continue
        if p < min_prime:  # factor floor (keep n coprime to all test bases)
            continue
        if v2_cap is not None and _v2(d) >= v2_cap:  # force class A (see docstring)
            continue
        if not prove(p, dprimes):
            continue
        values.append(p)
        rows.append([p % Q for Q in Qs])
        logs.append(flog(p))
    residues = np.array(rows, dtype=np.int64).reshape(-1, len(Qs))
    return values, residues, np.asarray(logs, dtype=np.float64)


def _generate_shard(task):
    """Top-level (picklable) worker: prove every prime in one prefix-branch subtree.

    ``task`` is ``(factors, prefix_len, prefix_exps, min_prime, v2_cap, Qs)``: the worker
    fixes the exponents of the first ``prefix_len`` primes to ``prefix_exps`` and
    enumerates all choices for the remaining primes.  The union over all ``prefix_exps``
    is exactly ``_divisors``.  ``min_prime``/``v2_cap`` are the optional filters of
    :func:`_sweep`; ``Qs`` may contain the experimental strengthened 2-primary modulus.
    """
    factors, prefix_len, prefix_exps, min_prime, v2_cap, Qs = task
    d0 = 1
    used0: tuple[int, ...] = ()
    for (q, _), e in zip(factors[:prefix_len], prefix_exps):
        if e:
            d0 *= q ** e
            used0 += (q,)
    excluded = frozenset(q for q, _ in factors)
    return _sweep(_enumerate_subtree(factors, prefix_len, d0, used0), excluded, Qs,
                  min_prime, v2_cap)


def _choose_prefix_len(factors, n_tasks_target):
    """Smallest prefix length whose exponent product reaches ``n_tasks_target`` tasks
    (capped at the number of factors)."""
    prefix_len = 0
    n_tasks = 1
    while prefix_len < len(factors) and n_tasks < n_tasks_target:
        n_tasks *= factors[prefix_len][1] + 1
        prefix_len += 1
    return prefix_len


def flog(n: int) -> float:
    """Natural log of a possibly-huge integer (``math.log`` overflows past ~2**1024)."""
    if n <= 0:
        raise ValueError("flog domain")
    bits = n.bit_length()
    if bits <= 1000:
        return math.log(n)
    shift = bits - 900
    return math.log(n >> shift) + shift * math.log(2)


@dataclass
class PrimeSet:
    """The set ``P`` in the representation used by Algorithm 1.

    Parallel arrays, all indexed by ``0 .. N-1``:

    * ``values[j]``    -- the prime ``p_j`` (exact Python int)
    * ``residues``     -- ``int64`` array of shape ``(N, r)``; ``residues[j, k] = p_j mod Q_k``
    * ``logs[j]``      -- ``ln p_j``

    plus the originating :class:`Lambda` and, for the push-down step, a per-column
    table of modular inverses.
    """

    lam: Lambda
    values: list[int]
    residues: np.ndarray  # (N, r) int64
    logs: np.ndarray  # (N,) float64
    pushdown_moduli: tuple[int, ...] | None = None

    @property
    def N(self) -> int:
        return len(self.values)

    @property
    def stage_moduli(self) -> tuple[int, ...]:
        """The pairwise-coprime moduli used by the push-down stages.

        Ordinarily these are exactly the prime-power factors of ``Lambda``.  The
        experimental class-A construction may strengthen only the 2-primary
        modulus while leaving ``Lambda``--and therefore the admissible-prime
        condition ``p-1 | Lambda``--unchanged.
        """
        return self.pushdown_moduli or self.lam.prime_powers

    def inverse_tables(self) -> list[np.ndarray]:
        """For each column ``k`` return an ``int64`` table ``inv`` of length ``Q_k`` with
        ``inv[x]`` = the inverse of ``x`` mod ``Q_k`` (``inv[x] = 0`` when ``x`` is not a unit).

        Every residue that occurs in column ``k`` is a unit mod ``Q_k`` (the primes are
        coprime to ``Lambda``), so the table is always defined where we index it.
        """
        tables = []
        for Q in self.stage_moduli:
            t = np.zeros(Q, dtype=np.int64)
            for x in range(1, Q):
                if math.gcd(x, Q) == 1:
                    t[x] = pow(x, -1, Q)
            tables.append(t)
        return tables


# Below this many divisors the process-pool overhead is not worth it; run serially.
_SERIAL_DIVISOR_THRESHOLD = 200_000


def _finalize(lam: Lambda, values, residues, logs,
              pushdown_moduli: tuple[int, ...] | None = None) -> PrimeSet:
    """Sort ``P`` into a canonical, scheduling-independent order and box it.

    Order is lexicographic on the residue vector (an ``int64`` C sort -- no big-integer
    comparisons), which is a total order in practice since distinct primes ``< Lambda``
    have distinct residue vectors.  The push-down re-sorts every stage, so this matters
    only for reproducibility, not correctness.
    """
    if len(values) > 1:
        order = np.lexsort(residues.T[::-1])  # column 0 (largest Q) most significant
        residues = residues[order]
        logs = logs[order]
        values = [values[i] for i in order.tolist()]
    return PrimeSet(lam=lam, values=values, residues=residues, logs=logs,
                    pushdown_moduli=pushdown_moduli)


def _with_two_adic_multiplier(lam: Lambda, multiplier: int) -> tuple[int, ...]:
    """Return stage moduli with the 2-primary column multiplied by ``multiplier``.

    ``multiplier`` must itself be a positive power of two.  Column order is
    deliberately preserved: this changes the one existing 2-primary stage rather
    than silently changing the push-down schedule.  All current project presets
    already put that stage first.
    """
    multiplier = int(multiplier)
    if multiplier < 1 or multiplier & (multiplier - 1):
        raise ValueError("two_adic_multiplier must be a positive power of two")
    if multiplier > 1 and 2 not in lam.prime_set:
        raise ValueError("cannot strengthen the 2-primary stage: 2 does not divide Lambda")
    return tuple(Q * multiplier if q == 2 else Q
                 for (q, _), Q in zip(lam.factors, lam.prime_powers))


def generate_prime_set(lam: Lambda, *, jobs=None, progress=None,
                       min_prime: int = 1, class_a: bool = False,
                       two_adic_multiplier: int = 1) -> PrimeSet:
    """Construct ``P`` for the modulus ``lam`` with rigorous primality proofs.

    ``jobs`` controls parallelism (default ``min(os.cpu_count(), 72)``); ``jobs=1`` forces
    the serial path.  ``progress`` is an optional ``callable(done, total, found)``.

    ``min_prime`` keeps only primes ``p >= min_prime``; ``class_a=True`` additionally drops
    primes with ``v_2(p-1) == v_2(Lambda)`` so every subset product is a class-A Carmichael.
    Both are used by the original Euler-pseudoprime construction (:mod:`pdc.euler`); see
    :func:`_sweep`.

    ``two_adic_multiplier=2`` or ``4`` selects a strengthened-modulus alternative:
    retain the complete original prime pool but replace its 2-primary push-down modulus
    ``Q`` by ``2Q`` or ``4Q``.  A full push-down then forces
    ``v_2(n-1) > max_{p|n} v_2(p-1)``, hence class A.  It is intentionally mutually
    exclusive with the old ``class_a`` prime filter so the two strategies can be compared.
    """
    factors = list(lam.factors)
    excluded = lam.prime_set
    Qs = _with_two_adic_multiplier(lam, two_adic_multiplier)
    if class_a and two_adic_multiplier != 1:
        raise ValueError("choose either the class_a prime filter or a stronger 2-adic "
                         "push-down modulus, not both")
    v2_cap = _v2(lam.value) if class_a else None

    total_div = 1
    for _, h in factors:
        total_div *= h + 1
    if jobs is None:
        jobs = min(os.cpu_count() or 1, 72)

    # Serial path: tiny moduli, or explicitly single-job.  Reports per-50k candidates.
    if jobs == 1 or total_div < _SERIAL_DIVISOR_THRESHOLD:
        if progress is None:
            values, residues, logs = _sweep(_enumerate_subtree(factors, 0, 1, ()), excluded,
                                            Qs, min_prime, v2_cap)
        else:
            values, rows, logs_l, checked = [], [], [], 0
            prove = proves_prime
            for d, dprimes in _enumerate_subtree(factors, 0, 1, ()):
                checked += 1
                if checked % 50000 == 0:
                    progress(checked, total_div, len(values))
                p = d + 1
                if p in excluded or p < min_prime:
                    continue
                if v2_cap is not None and _v2(d) >= v2_cap:
                    continue
                if not prove(p, dprimes):
                    continue
                values.append(p)
                rows.append([p % Q for Q in Qs])
                logs_l.append(flog(p))
            residues = np.array(rows, dtype=np.int64).reshape(-1, len(Qs))
            logs = np.asarray(logs_l, dtype=np.float64)
        return _finalize(lam, values, residues, logs, Qs)

    # Parallel path: partition the divisor recursion into independent prefix-branch
    # subtrees and prove each shard in its own process.
    prefix_len = _choose_prefix_len(factors, 8 * jobs)
    prefix_iter = itertools.product(*(range(h + 1) for _, h in factors[:prefix_len]))
    tasks = [(factors, prefix_len, pe, min_prime, v2_cap, Qs) for pe in prefix_iter]

    values: list[int] = []
    res_parts: list[np.ndarray] = []
    logs_parts: list[np.ndarray] = []
    done = 0
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for v, res, lg in ex.map(_generate_shard, tasks):
            values.extend(v)
            if res.size:
                res_parts.append(res)
                logs_parts.append(lg)
            done += 1
            if progress is not None:
                progress(done, len(tasks), len(values))

    r = lam.r
    residues = np.concatenate(res_parts, axis=0) if res_parts else np.empty((0, r), np.int64)
    logs = np.concatenate(logs_parts) if logs_parts else np.empty(0, np.float64)
    return _finalize(lam, values, residues, logs, Qs)
