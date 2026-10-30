"""Algorithm 1 of the paper -- "Many Carmichaels subset-product".

This is a faithful, modern re-implementation of the C program ``pushdown.c``.

Idea
----
We ordinarily work in ``G = (Z/Lambda Z)^*`` and want products of primes from ``P`` that
equal the identity.  Represent each element by its residue vector
``(x mod Q_0, ..., x mod Q_{r-1})`` with the columns ordered by *descending* ``Q_i`` (see
``lam.py``).  An element is "done through a prefix of length L" when its first ``L``
residues are all ``1``.  An experimental class-A variant may strengthen only the
2-primary stage modulus; the admissible primes still satisfy ``p-1 | Lambda``.

We sweep the columns ``m = 0, 1, ..., r-1`` (largest prime power first).  At stage ``m``
every surviving element is already ``== 1`` on columns ``0..m-1``.  We:

1. **Fix the global product.** Remove (discard) at most one element so that the product
   of all survivors is ``== 1 (mod Q_m)``.  This is the "throw away one number" step.
2. **Push down.** Greedily pair the remaining elements: for each ``a`` find ``a_hat``
   maximizing the agreement of ``a * a_hat`` with the identity.  Because the working set
   is kept sorted lexicographically by residue vector, the best partner for ``a`` is an
   immediate neighbor of ``a^{-1}`` in that order (this is the efficient match described
   in the paper's "Construction" section).  If the product clears column ``m`` (prefix
   grows past ``m``) the pair is merged and carried forward.
3. **Sweep the chaff.** Whatever could not be paired all multiplies together to a single
   element that is ``== 1`` on columns ``0..m`` (guaranteed by step 1); carry it forward.

After the last stage every survivor is ``== 1`` modulo every stage modulus, hence in
particular ``== 1 (mod Lambda)``: a "base" Carmichael number, recorded by the *history*
(the multiset of prime indices) that produced it.  The histories are pairwise disjoint, so
the product of any subset of base numbers is again ``== 1`` and hence Carmichael -- which
is what makes the factor-count coverage in ``coverage.py`` work.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .lam import Lambda
from .primes import PrimeSet

# Optional Cython hot loop.  Build it with ``python3 setup.py build_ext --inplace``
# (or ``pip install -e .``); without it we transparently use the pure-Python matcher,
# which produces identical pairings.
try:
    from . import _match as _cy_match  # type: ignore
    HAVE_CYTHON = True
except ImportError:  # pragma: no cover - exercised only when the extension isn't built
    _cy_match = None
    HAVE_CYTHON = False


@dataclass
class StageStat:
    Q: int           # the prime power Q_m pushed down at this stage
    pushed: int      # number of successful pair merges
    thrown: int      # elements discarded (0 or 1 normally; >0 chaff only if unfixable)
    thrown_factors: int
    chaff: int       # number of leftover elements combined into one
    remaining: int   # size of the working set after the stage


@dataclass
class PushdownResult:
    lam: Lambda
    primes: PrimeSet
    base: list[list[int]]          # each entry: the prime *indices* of one base Carmichael
    stages: list[StageStat] = field(default_factory=list)

    @property
    def factor_counts(self) -> list[int]:
        return [len(h) for h in self.base]


def _modprod_rows(rows: np.ndarray, Q_pd, dtype) -> np.ndarray:
    """Product of residue ``rows``, per column, reduced mod ``Q`` -> ``int64`` shape ``(r,)``.

    Reduces pairwise in a tree so every intermediate product is just two residues
    multiplied (``< Q**2``) before the modulo, keeping the arithmetic exact in ``int64``
    (or Python objects for an extreme ``Q``).
    """
    r = rows.shape[1]
    if rows.shape[0] == 0:
        return np.ones(r, dtype=np.int64)
    block = rows.astype(dtype)
    while block.shape[0] > 1:
        n = block.shape[0]
        h = n // 2
        paired = (block[:2 * h:2] * block[1:2 * h:2]) % Q_pd   # (h, r)
        if n % 2:
            paired = np.vstack([paired, block[-1:]])
        block = paired
    return (block[0] % Q_pd).astype(np.int64)


def _cpl(row: np.ndarray, target: tuple[int, ...]) -> int:
    """Common-prefix length of a residue row with a target residue tuple."""
    n = len(target)
    for k in range(n):
        if int(row[k]) != target[k]:
            return k
    return n


def push_down(primes: PrimeSet, *, max_stage: int | None = None, log=None,
              jobs: int | None = 1) -> PushdownResult:
    """Run Algorithm 1 over the prime set ``primes``.

    Parameters
    ----------
    max_stage:
        Push down only columns ``0 .. max_stage`` (default: all ``r`` columns -- a full
        run, which is required for the survivors to be Carmichael).  Exposed mainly for
        experimentation, mirroring the optional argument of ``pushdown.c``.
    log:
        Optional ``callable(str)`` for progress lines.
    """
    lam = primes.lam
    Qs = primes.stage_moduli
    r = len(Qs)
    N = primes.N
    inv_tables = primes.inverse_tables()
    last = r - 1 if max_stage is None else min(max_stage, r - 1)

    # Working set: residue matrix, plus parallel logs.  Histories are NOT carried as
    # explicit index lists (that is O(N*r) copying at scale); instead each working element
    # holds a `node` id into a merge forest, flattened to prime indices once at the end.
    res = primes.residues.copy()                      # (M, r) int64
    node = np.arange(N, dtype=np.int64)               # leaves 0..N-1 are prime indices
    hlen = np.ones(N, dtype=np.int64)                 # factor count carried by each element
    logs = primes.logs.copy()

    # Merge forest: internal node ids run N, N+1, ... in creation order.  A *pair* node
    # stores its two children in ileft/iright; the rare *chaff* node (one per stage) sets
    # iright = -1 and stores its many children in `chaff_children`.
    left_blocks: list[np.ndarray] = []
    right_blocks: list[np.ndarray] = []
    chaff_children: dict[int, np.ndarray] = {}
    next_id = N

    # For the vectorized residue products: a pairwise product of two residues is
    # ``< Q_k**2``.  That stays inside int64 for any realistic Lambda; for an extreme
    # Q we fall back to Python-object arithmetic, so the result is always exact.
    Qrow = np.asarray(Qs, dtype=np.int64)
    prod_dtype = np.int64 if (int(Qrow.max()) - 1) ** 2 < (1 << 62) else object
    Qrow_pd = Qrow.astype(object) if prod_dtype is object else Qrow

    result = PushdownResult(lam=lam, primes=primes, base=[])

    for m in range(last + 1):
        Q = Qs[m]
        col = res[:, m]
        pass_mask = col == 1
        cand_idx = np.flatnonzero(~pass_mask)

        # --- step 1: make the global product == 1 (mod Q_m) -------------------
        # Pass-through elements are == 1 on column m, so the product over the candidates
        # equals the product over the whole working set: reduce the column (vectorized).
        prod_m = int(_modprod_rows(col.reshape(-1, 1), Qrow_pd[m:m + 1], prod_dtype)[0])

        discarded = -1
        fixable = True
        thrown = 0
        thrown_factors = 0
        if prod_m != 1:
            # Tie-break target: the product of the whole working set, per column
            # (the `prod[]` vector of pushdown.c).  Discard the element that best matches
            # it, so removing it nudges the remaining product toward 1.  Only candidates
            # whose column-m residue equals prod_m can repair the product when removed.
            full = _modprod_rows(res, Qrow_pd, prod_dtype)        # (r,) int64
            sel = cand_idx[res[cand_idx, m] == prod_m]
            if sel.size == 0:
                fixable = False  # cannot repair the product at this column
            else:
                rows = res[sel]
                mism = rows != full
                cpl = np.where(mism.any(axis=1), mism.argmax(axis=1), r)  # common-prefix len
                # maximize agreement (cpl), then fewest factors (hlen)
                pick = np.lexsort((hlen[sel], -cpl))[0]
                discarded = int(sel[pick])
                thrown = 1
                thrown_factors = int(hlen[discarded])

        # candidate elements to be matched (exclude the discarded one)
        if discarded >= 0:
            cand_arr = cand_idx[cand_idx != discarded]
        else:
            cand_arr = cand_idx
        C = int(cand_arr.size)

        # --- step 2: greedy push-down via lexicographic neighbor match --------
        cres = res[cand_arr]                               # (C, r) int64
        # Row-wise modular inverses = the match targets (vectorized table lookups).
        cinv = np.empty_like(cres)
        for k in range(r):
            cinv[:, k] = inv_tables[k][cres[:, k]]
        # Lexicographic order of rows, column m most significant.  Columns 0..m-1 are
        # already == 1 on every candidate (push-down invariant), so they are constant sort
        # keys -- drop them and sort only columns m..r-1.  Identical order, m fewer passes.
        order = (np.lexsort(cres[:, m:].T[::-1]).astype(np.int64) if C
                 else np.empty(0, dtype=np.int64))

        partner, consumed = _match_stage(cres, cinv, order, m)
        consumed = np.asarray(consumed, dtype=bool)

        res_blocks: list[np.ndarray] = []     # (*, r) int64 residue blocks
        node_blocks: list[np.ndarray] = []    # parallel node ids
        hlen_blocks: list[np.ndarray] = []    # parallel factor counts
        log_blocks: list[np.ndarray] = []     # parallel log arrays

        # pass-through: elements already == 1 on column m keep their prefix, carried as-is
        pass_idx = np.flatnonzero(pass_mask)
        res_blocks.append(res[pass_idx])
        node_blocks.append(node[pass_idx])
        hlen_blocks.append(hlen[pass_idx])
        log_blocks.append(logs[pass_idx])

        # --- step 2 cont.: realize the merged pairs as one block --------------
        owners = np.flatnonzero(partner >= 0)          # cand-local indices owning a merge
        mates = partner[owners]
        pushed = int(owners.size)
        merged_block = ((cres[owners].astype(prod_dtype) * cres[mates].astype(prod_dtype))
                        % Qrow_pd).astype(np.int64).reshape(-1, r)
        own_w = cand_arr[owners]
        mate_w = cand_arr[mates]
        res_blocks.append(merged_block)
        node_blocks.append(np.arange(next_id, next_id + pushed, dtype=np.int64))
        hlen_blocks.append(hlen[own_w] + hlen[mate_w])
        log_blocks.append(logs[own_w] + logs[mate_w])
        left_blocks.append(node[own_w])
        right_blocks.append(node[mate_w])
        next_id += pushed

        # --- step 3: sweep the chaff -----------------------------------------
        chaff_idx = np.flatnonzero(~consumed)
        n_chaff = int(chaff_idx.size)
        if n_chaff:
            chaff_w = cand_arr[chaff_idx]
            if fixable:
                # product of all chaff is == 1 on columns 0..m, so it carries forward as a
                # single new node whose children are all the chaff elements' nodes
                acc = _modprod_rows(cres[chaff_idx], Qrow_pd, prod_dtype)
                res_blocks.append(acc.reshape(1, r))
                node_blocks.append(np.array([next_id], dtype=np.int64))
                hlen_blocks.append(np.array([int(hlen[chaff_w].sum())], dtype=np.int64))
                log_blocks.append(np.array([float(logs[chaff_w].sum())]))
                chaff_children[next_id] = node[chaff_w]
                left_blocks.append(np.array([-1], dtype=np.int64))   # sentinel: chaff node
                right_blocks.append(np.array([-1], dtype=np.int64))
                next_id += 1
            else:
                # product was unfixable: discard the chaff entirely (as pushdown.c does)
                thrown = n_chaff
                thrown_factors = int(hlen[chaff_w].sum())

        # --- assemble the working set for the next stage ----------------------
        res = np.vstack(res_blocks).reshape(-1, r).astype(np.int64, copy=False)
        node = np.concatenate(node_blocks)
        hlen = np.concatenate(hlen_blocks)
        logs = np.concatenate(log_blocks) if log_blocks else np.empty(0, np.float64)

        result.stages.append(
            StageStat(Q=Q, pushed=pushed, thrown=thrown, thrown_factors=thrown_factors,
                      chaff=n_chaff, remaining=int(node.size))
        )
        if log is not None:
            log(f"Pushed down on {Q}: merged {pushed}, threw {thrown} "
                f"({thrown_factors} factors), {int(node.size)} remaining")

    # --- flatten the merge forest: each surviving root -> its prime-index list -----
    result.base = _flatten_forest(node, left_blocks, right_blocks, chaff_children, N, jobs=jobs)
    return result


def _flatten_one(roots_list, ileft, iright, chaff_children, N) -> list[list[int]]:
    """Iterative-DFS leaf collection for a list of roots (the per-root worker body)."""
    base: list[list[int]] = []
    for rt in roots_list:
        out: list[int] = []
        stack = [rt]
        while stack:
            nid = stack.pop()
            if nid < N:
                out.append(nid)
            elif iright[nid - N] == -1:
                stack.extend(int(c) for c in chaff_children[nid])
            else:
                stack.append(int(ileft[nid - N]))
                stack.append(int(iright[nid - N]))
        base.append(out)
    return base


# Fork-inherited state for the parallel forest flatten (avoids pickling the child arrays).
_FLAT_G: dict = {}


def _flatten_chunk(roots_list):
    g = _FLAT_G
    return _flatten_one(roots_list, g["ileft"], g["iright"], g["chaff_children"], g["N"])


def _flatten_forest(roots, left_blocks, right_blocks, chaff_children, N,
                    jobs: int | None = 1):
    """Collect the leaf (prime) indices under each surviving root node.

    Pair nodes store children in ``ileft``/``iright``; a chaff node (``iright == -1``)
    stores its children in ``chaff_children``.  Iterative DFS, O(total leaves) = O(N).
    Roots are independent, so with ``jobs > 1`` (or ``None`` -> auto) the roots are sharded
    across worker processes that fork-inherit the child arrays (mirrors ``euler._ROW_G``).
    """
    ileft = np.concatenate(left_blocks) if left_blocks else np.empty(0, np.int64)
    iright = np.concatenate(right_blocks) if right_blocks else np.empty(0, np.int64)
    roots_list = roots.tolist()
    if jobs is None:
        import os
        jobs = min(os.cpu_count() or 1, 72)
    if jobs <= 1 or len(roots_list) < 4096:
        return _flatten_one(roots_list, ileft, iright, chaff_children, N)

    from concurrent.futures import ProcessPoolExecutor
    _FLAT_G.update(ileft=ileft, iright=iright, chaff_children=chaff_children, N=N)
    step = max(1, len(roots_list) // (8 * jobs))
    chunks = [roots_list[i:i + step] for i in range(0, len(roots_list), step)]
    base: list[list[int]] = []
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for part in ex.map(_flatten_chunk, chunks):
            base.extend(part)
    return base


def _match_stage(cres: np.ndarray, cinv: np.ndarray, order: np.ndarray, m: int):
    """Decide the pairings for one push-down stage.

    Returns ``(partner, consumed)``: ``partner[a] = b >= 0`` means ``a`` owns the merged
    pair ``(a, b)`` (and ``b`` is consumed); ``partner[a] = -1`` means ``a`` is chaff or
    was consumed.  Dispatches to the Cython kernel when available, otherwise to an
    identical pure-Python implementation.
    """
    if len(cres) == 0:
        return np.empty(0, np.int64), np.zeros(0, bool)
    if HAVE_CYTHON:
        return _cy_match.match_stage(
            np.ascontiguousarray(cres), np.ascontiguousarray(cinv),
            np.ascontiguousarray(order), m)
    return _match_stage_py(cres, cinv, order, m)


def _match_stage_py(cres: np.ndarray, cinv: np.ndarray, order: np.ndarray, m: int):
    """Pure-Python reference for :func:`_match_stage` (the Cython fallback).

    For each not-yet-consumed candidate, lexicographically binary-search its inverse
    vector into the sorted residue order, take the first available neighbor on each side,
    and pair with the better one when the common-prefix length exceeds ``m``.
    """
    C, r = cres.shape
    order_list = order.tolist()
    sorted_rows = [cres[o] for o in order_list]
    consumed = np.zeros(C, dtype=bool)
    partner = np.full(C, -1, dtype=np.int64)

    def row_ge_key(row, key) -> bool:
        for k in range(r):
            if row[k] < key[k]:
                return False
            if row[k] > key[k]:
                return True
        return True

    for ai in range(C):
        if consumed[ai]:
            continue
        key = cinv[ai]
        lo, hi = 0, C
        while lo < hi:
            mid = (lo + hi) >> 1
            if row_ge_key(sorted_rows[mid], key):
                hi = mid
            else:
                lo = mid + 1
        pos = lo

        left = -1
        j = pos - 1
        while j >= 0:
            oi = order_list[j]
            if oi != ai and not consumed[oi]:
                left = oi
                break
            j -= 1

        right = -1
        j = pos
        while j < C:
            oi = order_list[j]
            if oi != ai and not consumed[oi]:
                right = oi
                break
            j += 1

        if left == -1:
            best = right
        elif right == -1:
            best = left
        else:
            best = right if _cpl(cres[right], key) > _cpl(cres[left], key) else left

        if best != -1 and _cpl(cres[best], key) > m:
            consumed[ai] = True
            consumed[best] = True
            partner[ai] = best
    return partner, consumed
