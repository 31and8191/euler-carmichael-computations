"""Construct Euler pseudoprimes to many prime bases, by F2 linear algebra over
class-A base Carmichael numbers.

Background
----------
The Solovay-Strassen test declares ``n`` prime for base ``a`` when the *Euler condition*

    a ** ((n - 1) / 2)  ==  (a / n)   (mod n)          (a Jacobi symbol on the right)

holds; a composite ``n`` that passes is an *Euler pseudoprime* to base ``a``.

For a Carmichael number ``n`` in **class A** -- i.e. with even index ``i(n) = (n-1)/lambda(n)``,
equivalently ``v_2(n-1) > max_{p|n} v_2(p-1)`` -- one has ``a ** ((n-1)/2) == 1 (mod n)`` for
*every* ``a`` coprime to ``n`` (Alcantarilla Sanchez--Cottaar--Lange--de Weger, "Ours goes to
211", Lemma on class A).  Hence

    n (class A) is an Euler pseudoprime to base a   <=>   (a / n) = +1 .

Now take the pairwise-coprime base Carmichael numbers ``N_1, ..., N_m`` that Algorithm 1
produces from a single modulus ``Lambda`` (see :mod:`pdc.pushdown`).  There are two
supported ways to force every output into class A:

  * the original ``class_a=True`` prime filter drops primes with
    ``v_2(p-1) = v_2(Lambda)``; or
  * the experimental Shallue-style path retains the full ``p-1 | Lambda`` pool and uses
    ``two_adic_multiplier=2`` or ``4`` to push down modulo ``2Q`` or ``4Q`` at the
    2-primary stage.

Either gives ``v_2(n-1) > max v_2(p-1)`` for every output.  Independently,
``min_prime > p_t`` keeps every output coprime to all test bases.

Because the Jacobi symbol is completely multiplicative in its "denominator",

    (a / prod_{j in S} N_j)  =  prod_{j in S} (a / N_j),

the *bit* ``x_{a,j} = [ (a/N_j) = -1 ]`` is linear over ``F_2`` in the selection
``sigma in F_2^m``:  ``[ (a/n) = -1 ] = sum_{j in S} x_{a,j} (mod 2)``.  So

    n = prod_{j : sigma_j = 1} N_j  is an Euler pseudoprime to base a   <=>   <x_a, sigma> = 0 .

Collecting one row ``x_a`` per test base ``a`` gives a matrix ``M`` over ``F_2``; any nonzero
vector in ``ker M`` is a subset product that is simultaneously an Euler pseudoprime to *all*
the target bases.  With ``m`` (number of coprime base Carmichaels) exceeding the number of
constrained bases, such a kernel vector is guaranteed to exist -- we *solve* for it rather
than hoping a random product survives (the improvement over pairwise multiplication).

Two simplifications:

  * An odd base ``q`` dividing ``Lambda`` is automatic and imposes no constraint (it never
    enters ``M``): with ``n == 1 (mod 4)`` and ``n == 1 (mod q)``, quadratic reciprocity
    gives ``(q/n) = (n/q) = (1/q) = +1``.  For ``q = 2`` reciprocity does not apply:
    ``(2/n) = +1`` iff ``n == +-1 (mod 8)``.  Base 2 is automatic when the enforced
    2-primary stage modulus is divisible by 8; otherwise it is retained as an ordinary
    row of ``M``.  This separates the class-A requirement from the base-2 choice.
  * Euler liars form a subgroup of ``(Z/nZ)^*``, so if the first ``t`` primes are all liars
    then so is every integer below the next prime -- "survives the first ``t`` prime bases".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .lam import Lambda
from .pushdown import PushdownResult


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #
def _v2(n: int) -> int:
    """2-adic valuation of a nonzero integer."""
    return (n & -n).bit_length() - 1


def first_primes(t: int) -> list[int]:
    """The first ``t`` primes (2, 3, 5, ...)."""
    if t <= 0:
        return []
    # Rosser bound on the t-th prime, plus slack; sieve up to there.
    import math
    if t < 6:
        limit = 15
    else:
        # Rosser's bound: the t-th prime is < t (ln t + ln ln t) for t >= 6.
        limit = int(t * (math.log(t) + math.log(math.log(t)))) + 10
    while True:
        sieve = np.ones(limit + 1, dtype=bool)
        sieve[:2] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                sieve[i * i::i] = False
        primes = np.flatnonzero(sieve).tolist()
        if len(primes) >= t:
            return primes[:t]
        limit *= 2  # bound was tight (should not happen for t >= 6); grow and retry


def _qr_table(q: int) -> np.ndarray:
    """``t[r] = +1`` if ``r`` is a nonzero quadratic residue mod prime ``q``, else ``-1``
    (``t[0] = 0``).  Length ``q``."""
    t = np.full(q, -1, dtype=np.int8)
    x = np.arange(1, q, dtype=np.int64)
    t[(x * x) % q] = 1        # vectorized: mark every nonzero quadratic residue
    t[0] = 0
    return t


# --------------------------------------------------------------------------- #
# Legendre / Jacobi bits (vectorized over the whole prime set)
# --------------------------------------------------------------------------- #
def _legendre_bits(base: int, limbs: list, p_mod8: np.ndarray) -> np.ndarray:
    """For a prime ``base`` and every prime ``p`` return ``uint8`` bits: ``1`` iff the
    Legendre symbol ``(base / p) = -1``.

    ``limbs`` is the primes in 64-bit-limb form (``limbs[i][j]`` = 64-bit limb ``i`` of
    ``p_j``); ``p_mod8`` is ``p mod 8`` (int8).  The primes ``p`` all exceed ``base``, so
    ``base`` is a unit mod ``p`` and the symbol is never ``0``.
    """
    if base == 2:
        # (2/p) = +1 iff p == +-1 (mod 8)
        return ((p_mod8 == 3) | (p_mod8 == 5)).astype(np.uint8)
    # p mod base, fully vectorized in int64 via limbs: p = sum_i limb_i * 2^(64 i), so
    # p mod base = (sum_i (limb_i mod base) * (2^(64 i) mod base)) mod base.  Each term is
    # < base^2 < 2^42, reduced every step -> no object-dtype mod, no overflow.
    pq = np.zeros(limbs[0].shape[0], dtype=np.int64)
    for i, limb in enumerate(limbs):
        c = pow(2, 64 * i, base)                          # scalar, < base
        pq = (pq + (limb % base).astype(np.int64) * c) % base
    # (base/p) via reciprocity: (base/p) = (p/base) * eps, eps = -1 iff base==3==p (mod 4)
    leg = _qr_table(base)[pq]                              # (p / base) as +-1
    if base % 4 == 3:
        leg = np.where((p_mod8 & 3) == 3, -leg, leg)       # p == 3 (mod 4) -> flip sign
    return (leg == -1).astype(np.uint8)


# --------------------------------------------------------------------------- #
# F2 kernel
# --------------------------------------------------------------------------- #
def f2_kernel_vector(rows: list[int], ncols: int, *, want_sparse: bool = False,
                     sparse_scan: int = 4096) -> int | None:
    """Return a nonzero vector ``sigma`` (as an integer bitset, ``bit j`` = column ``j``)
    with ``M sigma = 0`` over ``F_2``, where ``M`` is the matrix whose rows are ``rows``
    (each an integer bitset over ``ncols`` columns).  Returns ``None`` if the only solution
    is ``sigma = 0`` (the constraints already pin every column).

    Reduces ``M`` to reduced row echelon form using the highest set bit of each row as its
    pivot, then reads a kernel basis vector off a free (non-pivot) column.  With
    ``want_sparse`` it scans up to ``sparse_scan`` free columns and returns the lowest-weight
    kernel vector found (fewer base Carmichaels -> smaller ``n``); otherwise it returns the
    vector of the first free column (cheap).
    """
    pivots = _rref(rows)
    return _kernel_from_pivots(pivots, ncols, want_sparse=want_sparse, sparse_scan=sparse_scan)


def _rref(rows) -> dict[int, int]:
    """Reduced row echelon form of the ``F_2`` matrix whose rows are integer bitsets.

    Returns ``{pivot_column: reduced_row}`` where each pivot column is the highest set bit
    of its row.  Invariant: distinct pivot columns, and every pivot row is 0 in every
    *other* pivot column -- so a pivot row expresses its pivot variable purely in terms of
    free variables, which is what :func:`_kernel_from_pivots` relies on.
    """
    pivots: dict[int, int] = {}
    for r in rows:
        cur = r
        for pc, prow in pivots.items():        # full reduction against all pivots
            if (cur >> pc) & 1:
                cur ^= prow
        if cur == 0:
            continue
        c = cur.bit_length() - 1               # new pivot column
        for pc in pivots:                      # clear column c from existing pivot rows
            if (pivots[pc] >> c) & 1:
                pivots[pc] ^= cur
        pivots[c] = cur
    return pivots


def _kernel_from_pivots(pivots: dict[int, int], ncols: int, *, want_sparse: bool = False,
                        sparse_scan: int = 4096) -> int | None:
    """Read a nonzero kernel basis vector off a free (non-pivot) column of an RREF."""
    pivot_cols = set(pivots)
    if len(pivot_cols) >= ncols:
        return None

    def kernel_of_free(f: int) -> int:
        e = 1 << f
        for pc, prow in pivots.items():
            if (prow >> f) & 1:
                e |= 1 << pc
        return e

    free_cols = (f for f in range(ncols) if f not in pivot_cols)
    if not want_sparse:
        return kernel_of_free(next(free_cols))

    best = None
    best_w = None
    for i, f in enumerate(free_cols):
        if i >= sparse_scan:
            break
        e = kernel_of_free(f)
        w = bin(e).count("1")
        if best_w is None or w < best_w:
            best, best_w = e, w
            if w == 1:
                break
    return best


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #
@dataclass
class EulerResult:
    """The outcome of :func:`construct_euler_pseudoprime`."""
    lam: Lambda
    target_bases: list[int]              # the first t primes we aimed to survive
    auto_bases: list[int]                # bases dividing Lambda (satisfied for free)
    hard_bases: list[int]                # bases not dividing Lambda (the rows of M)
    rank: int                            # rank of M over F2
    n_columns: int                       # number of base-Carmichael columns m
    selected: list[int] = field(default_factory=list)     # chosen column indices
    factor_values: list[int] = field(default_factory=list)  # the primes dividing n
    solved: bool = False

    @property
    def num_factors(self) -> int:
        return len(self.factor_values)

    def form_n(self) -> int:
        """The pseudoprime ``n`` itself (may be astronomically large)."""
        n = 1
        for p in self.factor_values:
            n *= p
        return n


def _column_index_arrays(result: PushdownResult):
    """Base-Carmichael columns as arrays of prime indices, plus the ``prime index -> column``
    map used to compute all column parities in one vectorized pass.

    Every push-down survivor is ``== 1 (mod Lambda)`` and squarefree with a disjoint factor
    set, so each is a valid pairwise-coprime building block; columns with a single prime
    factor cannot help form a >=3-factor product on their own but are harmless to include.
    """
    base = result.base
    N = result.primes.N
    col_id = np.full(N, -1, dtype=np.int64)
    cols = []
    for j, idx in enumerate(base):
        arr = np.asarray(idx, dtype=np.int64)
        cols.append(arr)
        col_id[arr] = j
    return cols, col_id


def _precompute(result: PushdownResult):
    """Column arrays + the per-prime data reused when building every matrix row."""
    if len(result.stages) != len(result.primes.stage_moduli):
        raise ValueError("Euler construction requires a complete push-down over every stage")
    two_modulus = _two_primary_stage_modulus(result)
    max_factor_v2 = max((_v2(p - 1) for p in result.primes.values), default=0)
    if _v2(two_modulus) <= max_factor_v2:
        raise ValueError(
            "push-down does not guarantee class A: "
            f"v_2(enforced 2-primary modulus)={_v2(two_modulus)} <= "
            f"max v_2(p-1)={max_factor_v2}")
    values = result.primes.values
    cols, col_id = _column_index_arrays(result)
    vals_obj = np.array(values, dtype=object)
    p_mod8 = (vals_obj % 8).astype(np.int8)
    valid = col_id >= 0
    # 64-bit limb decomposition of the (possibly >64-bit) primes, computed once so that
    # every row's `p mod q` is a vectorized int64 op instead of an object-dtype mod.
    n_limbs = ((result.lam.value + 1).bit_length() + 63) // 64
    mask64 = (1 << 64) - 1
    limbs = [((vals_obj >> (64 * i)) & mask64).astype(np.uint64) for i in range(n_limbs)]
    return values, cols, col_id, len(cols), vals_obj, p_mod8, valid, limbs


def _one_row(q, col_id, m, limbs, p_mod8, valid) -> int:
    bits = _legendre_bits(q, limbs, p_mod8)                  # (N,) uint8
    # per-column sum of the Legendre bits; bincount is ~10-50x faster than np.add.at here
    cnt = np.bincount(col_id[valid], weights=bits[valid], minlength=m).astype(np.int64)
    row_bits = (cnt & 1).astype(np.uint8)                    # parity = Jacobi bit
    return int.from_bytes(np.packbits(row_bits, bitorder="little").tobytes(), "little")


# Fork-inherited state for the parallel row builder (avoids pickling the big arrays).
_ROW_G: dict = {}


def _row_worker(chunk):
    g = _ROW_G
    return [_one_row(q, g["col_id"], g["m"], g["limbs"], g["p_mod8"], g["valid"])
            for q in chunk]


def _hard_rows(hard_bases, col_id, m, limbs, p_mod8, valid, log=None, jobs: int = 1
               ) -> list[int]:
    """One bit-packed F2 row per hard base: ``row[j] = [ (q / N_j) = -1 ]`` (parity over the
    factors of column ``N_j``).  Parallelized across processes for large runs (workers
    fork-inherit the shared arrays)."""
    if jobs <= 1 or len(hard_bases) < 128:
        rows: list[int] = []
        for i, q in enumerate(hard_bases):
            rows.append(_one_row(q, col_id, m, limbs, p_mod8, valid))
            if log is not None and (i + 1) % 100 == 0:
                log(f"  built {i + 1}/{len(hard_bases)} hard-base rows")
        return rows

    from concurrent.futures import ProcessPoolExecutor
    _ROW_G.update(col_id=col_id, m=m, limbs=limbs, p_mod8=p_mod8, valid=valid)
    chunk = max(1, len(hard_bases) // (8 * jobs))
    chunks = [hard_bases[i:i + chunk] for i in range(0, len(hard_bases), chunk)]
    rows = []
    done = 0
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for part in ex.map(_row_worker, chunks):
            rows.extend(part)
            done += len(part)
            if log is not None:
                log(f"  built {done}/{len(hard_bases)} hard-base rows")
    return rows


def _two_primary_stage_modulus(result: PushdownResult) -> int:
    """Return the enforced modulus in the column originating from ``2^s | Lambda``."""
    for (q, _), Q in zip(result.lam.factors, result.primes.stage_moduli):
        if q == 2:
            return Q
    raise ValueError("Euler construction requires 2 to divide Lambda")


def _base2_is_automatic(result: PushdownResult) -> bool:
    """Whether every push-down output is forced to be ``1 mod 8``."""
    return _two_primary_stage_modulus(result) % 8 == 0


def _split_bases(target_bases, lam_primes, *, base2_automatic: bool = True):
    auto = [q for q in target_bases
            if q in lam_primes and (q != 2 or base2_automatic)]
    hard = [q for q in target_bases
            if q not in lam_primes or (q == 2 and not base2_automatic)]
    return auto, hard


def _assemble(result, values, cols, m, target_bases, auto_bases, hard_bases, rows, sigma,
              rank=None):
    if rank is None:
        rank = _rank_f2(rows)
    eres = EulerResult(lam=result.lam, target_bases=target_bases, auto_bases=auto_bases,
                       hard_bases=hard_bases, rank=rank, n_columns=m)
    if sigma is None:
        eres.solved = False
        return eres
    selected = [j for j in range(m) if (sigma >> j) & 1]
    factor_idx: list[int] = []
    for j in selected:
        factor_idx.extend(int(i) for i in cols[j])
    eres.selected = selected
    eres.factor_values = sorted(values[i] for i in factor_idx)
    eres.solved = True
    if eres.factor_values:  # fail loudly rather than emit a wrong "pseudoprime"
        assert min(eres.factor_values) > (max(target_bases) if target_bases else 0), \
            "a factor of n is <= a test base: n would not be coprime to that base"
    return eres


def construct_euler_pseudoprime(result: PushdownResult, num_bases: int, *,
                                want_sparse: bool = False, jobs: int = 1, log=None
                                ) -> EulerResult:
    """Solve for a subset product of ``result``'s base Carmichaels that is an Euler
    pseudoprime to the first ``num_bases`` primes.

    ``result`` must have a complete push-down whose enforced 2-primary modulus has
    valuation strictly above every ``v_2(p-1)`` in the prime pool (so every output is class
    A), and ``min_prime`` must exceed the largest target base (so ``n`` is coprime to every
    base).  Both preconditions are checked.
    """
    values, cols, col_id, m, vals_obj, p_mod8, valid, limbs = _precompute(result)
    target_bases = first_primes(num_bases)
    auto_bases, hard_bases = _split_bases(
        target_bases, set(result.lam.primes),
        base2_automatic=_base2_is_automatic(result))
    rows = _hard_rows(hard_bases, col_id, m, limbs, p_mod8, valid, log, jobs)
    sigma = f2_kernel_vector(rows, m, want_sparse=want_sparse)
    return _assemble(result, values, cols, m, target_bases, auto_bases, hard_bases, rows, sigma)


def construct_maximal(result: PushdownResult, base_cap: int, *,
                      want_sparse: bool = False, jobs: int = 1, log=None) -> EulerResult:
    """Find the *largest* ``t <= base_cap`` for which a subset product is an Euler pseudoprime
    to the first ``t`` primes, and return that solution.

    Builds every hard-base row up to ``base_cap`` once, then walks the primes in order,
    accumulating rank: adding a hard base can only add a constraint, so solvability is
    monotone -- the boundary is the first hard base whose row pushes the rank to ``m`` (the
    number of coprime base Carmichaels), at which point the kernel becomes trivial.  If the
    rank never reaches ``m`` we are limited by ``base_cap`` (the factor floor), not by rank.
    """
    values, cols, col_id, m, vals_obj, p_mod8, valid, limbs = _precompute(result)
    full_target = first_primes(base_cap)
    lam_primes = set(result.lam.primes)
    base2_automatic = _base2_is_automatic(result)
    hard_positions = [i for i, q in enumerate(full_target)
                      if q not in lam_primes or (q == 2 and not base2_automatic)]
    hard_all = [full_target[i] for i in hard_positions]
    rows_all = _hard_rows(hard_all, col_id, m, limbs, p_mod8, valid, log, jobs)

    # Single incremental RREF.  Add hard rows in prime order; stop just before the row that
    # would push the rank to m (kernel would become trivial).  The pivots then hold the RREF
    # of exactly the kept prefix, so the kernel is read straight off them -- no second pass.
    pivots: dict[int, int] = {}
    h_star = len(rows_all)                       # floor-limited unless we break below
    for j, r in enumerate(rows_all):
        cur = r
        for pc, prow in pivots.items():
            if (cur >> pc) & 1:
                cur ^= prow
        if cur == 0:
            continue                             # dependent hard base: no new constraint
        if len(pivots) + 1 == m:                 # adding this row exhausts the kernel
            h_star = j
            break
        c = cur.bit_length() - 1
        for pc in pivots:
            if (pivots[pc] >> c) & 1:
                pivots[pc] ^= cur
        pivots[c] = cur

    if h_star >= len(rows_all):
        t_star = base_cap                        # floor-limited: every base below the floor works
    else:
        t_star = hard_positions[h_star]          # primes before the (h_star+1)-th hard base
    if log is not None:
        limit = "floor (base_cap)" if h_star >= len(rows_all) else "rank (kernel exhausted)"
        log(f"  maximal t = {t_star} (limited by {limit}); "
            f"rank {len(pivots)} of m={m} columns; {h_star} hard-base constraints used")

    target_bases = first_primes(t_star)
    auto_bases, hard_bases = _split_bases(
        target_bases, lam_primes, base2_automatic=base2_automatic)
    rows = rows_all[:h_star] if h_star < len(rows_all) else rows_all
    sigma = _kernel_from_pivots(pivots, m, want_sparse=want_sparse)
    return _assemble(result, values, cols, m, target_bases, auto_bases, hard_bases, rows, sigma,
                     rank=len(pivots))


def _rank_f2(rows: list[int]) -> int:
    """Rank over F2 of the matrix whose rows are the integer bitsets ``rows``."""
    pivots: dict[int, int] = {}
    for r in rows:
        cur = r
        while cur:
            c = cur.bit_length() - 1
            piv = pivots.get(c)
            if piv is None:
                pivots[c] = cur
                break
            cur ^= piv
    return len(pivots)


# --------------------------------------------------------------------------- #
# self-check (construction-level; the standalone gold-standard is tools/verify_euler.py)
# --------------------------------------------------------------------------- #
@dataclass
class StructuralCheck:
    """Cheap O(#factors) checks: is n a class-A Carmichael?  (No per-base work.)"""
    squarefree: bool
    at_least_3_factors: bool
    n_mod_lambda_is_one: bool
    class_a: bool
    v2_n_minus_1: int
    max_v2_p: int

    @property
    def is_class_a_carmichael(self) -> bool:
        return (self.squarefree and self.at_least_3_factors
                and self.n_mod_lambda_is_one and self.class_a)


def structural_check(eres: EulerResult) -> StructuralCheck:
    """Verify n is a class-A Carmichael from the factor list + Lambda alone, in one pass over
    the factors (no per-base loop -- use for large runs, then verify bases with
    :func:`verify_construction` or, at scale, ``tools/verify_euler.py``)."""
    lam_value = eres.lam.value
    factors = eres.factor_values
    k = len(factors)
    squarefree = len(set(factors)) == k
    at_least_3 = k >= 3

    n_mod_lambda = 1
    for p in factors:
        n_mod_lambda = (n_mod_lambda * (p % lam_value)) % lam_value

    M2 = (1 << 64) - 1
    n_low = 1
    for p in factors:
        n_low = (n_low * (p & M2)) & M2
    v2_n1 = _v2((n_low - 1) & M2) if n_low != 1 else 64
    max_v2_p = max((_v2(p - 1) for p in factors), default=0)
    return StructuralCheck(squarefree=squarefree, at_least_3_factors=at_least_3,
                           n_mod_lambda_is_one=(n_mod_lambda == 1),
                           class_a=(v2_n1 > max_v2_p), v2_n_minus_1=v2_n1, max_v2_p=max_v2_p)


@dataclass
class EulerCheck:
    ok: bool
    class_a: bool
    n_mod_lambda_is_one: bool
    squarefree: bool
    at_least_3_factors: bool
    failing_base: int | None      # first target base that is NOT an Euler liar, if any
    survived_bases: int           # count of leading target bases that ARE liars


def verify_construction(eres: EulerResult, *, gold_modexp_bit_limit: int = 700_000
                        ) -> EulerCheck:
    """Independent-of-the-kernel check that ``eres`` really is an Euler pseudoprime to all
    its target bases.

    Runs :func:`structural_check` (class-A Carmichael), then per target base ``a`` recomputes
    the Euler condition -- in class A that is exactly ``(a/n) = +1``, i.e. an even number of
    factors with ``(a/p) = -1``, computed here with the vectorized reciprocity path.  When
    ``n`` is small (``< gold_modexp_bit_limit`` bits) and ``gmpy2`` is available, it
    *additionally* runs the literal ``a**((n-1)//2) mod n`` against the Jacobi symbol as a
    gold-standard cross-check.

    Note: the per-base pass is serial over bases; for very large base counts prefer the
    parallel, construction-independent ``tools/verify_euler.py``.
    """
    factors = eres.factor_values
    sc = structural_check(eres)

    vals_obj = np.array(factors, dtype=object)
    p_mod8 = (vals_obj % 8).astype(np.int8)
    n_limbs = ((eres.lam.value + 1).bit_length() + 63) // 64
    mask64 = (1 << 64) - 1
    limbs = [((vals_obj >> (64 * i)) & mask64).astype(np.uint64) for i in range(n_limbs)]
    failing_base = None
    survived = 0
    for a in eres.target_bases:
        neg = int(_legendre_bits(a, limbs, p_mod8).sum()) & 1   # (a/n) = prod (a/p), bit
        liar = sc.class_a and (neg == 0)          # class A => Euler liar iff Jacobi == +1
        if liar:
            survived += 1
        else:
            failing_base = a
            break

    ok = sc.is_class_a_carmichael and failing_base is None

    # gold standard: literal Solovay-Strassen modexp on the actual integer, if small enough
    n_bits = int(sum(p.bit_length() for p in factors))  # ~ upper bound on bit length of n
    if ok and n_bits < gold_modexp_bit_limit and factors:
        try:
            import gmpy2
        except ModuleNotFoundError:
            gmpy2 = None
        if gmpy2 is not None:
            n = gmpy2.mpz(1)
            for p in factors:
                n *= p
            e = (n - 1) // 2
            for a in eres.target_bases:
                lhs = gmpy2.powmod(a, e, n)
                j = gmpy2.jacobi(a, n) % n       # +1 or n-1
                if lhs != j:
                    ok = False
                    failing_base = int(a)
                    break

    return EulerCheck(ok=ok, class_a=sc.class_a, n_mod_lambda_is_one=sc.n_mod_lambda_is_one,
                      squarefree=sc.squarefree, at_least_3_factors=sc.at_least_3_factors,
                      failing_base=failing_base, survived_bases=survived)
