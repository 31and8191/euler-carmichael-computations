# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Cython hot loop for the push-down stage matcher (see ``pushdown.py``).

This re-implements, directly on the ``int64`` residue matrix, the per-stage greedy
pairing that dominates Algorithm 1's running time:

  * for each not-yet-consumed candidate ``a``, lexicographically binary-search its
    inverse vector into the sorted residue order,
  * take the first available neighbor on each side (longest common prefix among
    available elements lies at the immediate sorted neighbor),
  * if the better neighbor clears the current column (common-prefix length ``> m``),
    pair them.

Everything here is integer-only and runs ``nogil``; the Python side
(``pushdown._match_stage``) handles histories and the modular product of paired rows.
The pure-Python fallback in ``pushdown.py`` produces identical pairings.
"""

import numpy as np
cimport numpy as cnp
from libc.stdint cimport int64_t

cnp.import_array()


cdef inline int row_ge_key(const int64_t[:, ::1] cres, Py_ssize_t row,
                           const int64_t[:, ::1] cinv, Py_ssize_t arow,
                           Py_ssize_t r) noexcept nogil:
    """1 iff residue row ``cres[row]`` >= key ``cinv[arow]`` lexicographically."""
    cdef Py_ssize_t k
    cdef int64_t a, b
    for k in range(r):
        a = cres[row, k]
        b = cinv[arow, k]
        if a < b:
            return 0
        elif a > b:
            return 1
    return 1  # equal counts as >=


cdef inline Py_ssize_t cpl_row_key(const int64_t[:, ::1] cres, Py_ssize_t row,
                                   const int64_t[:, ::1] cinv, Py_ssize_t arow,
                                   Py_ssize_t r) noexcept nogil:
    """Common-prefix length of ``cres[row]`` with key ``cinv[arow]``."""
    cdef Py_ssize_t k
    for k in range(r):
        if cres[row, k] != cinv[arow, k]:
            return k
    return r


def match_stage(int64_t[:, ::1] cres, int64_t[:, ::1] cinv,
                int64_t[::1] order, int m):
    """Return ``(partner, consumed)`` for one push-down stage.

    Parameters
    ----------
    cres : (C, r) int64
        Residue vectors of the candidate elements (those not yet ``== 1`` on column m).
    cinv : (C, r) int64
        Row-wise modular inverses of ``cres`` (the match targets).
    order : (C,) int64
        Indices sorting the rows of ``cres`` lexicographically (column 0 primary).
    m : int
        Current stage / column index; a pair is kept only if it clears column m.

    Returns
    -------
    partner : (C,) int64
        ``partner[a] = b >= 0`` means ``a`` is the owner of a merged pair ``(a, b)``;
        ``b`` is consumed.  ``partner[a] = -1`` means ``a`` is chaff or was consumed.
    consumed : (C,) bool
        Whether each candidate was absorbed (as owner or partner) during the stage.
    """
    cdef Py_ssize_t C = cres.shape[0]
    cdef Py_ssize_t r = cres.shape[1]

    cdef cnp.ndarray[cnp.int64_t, ndim=1] partner_arr = np.full(C, -1, dtype=np.int64)
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] consumed_arr = np.zeros(C, dtype=np.uint8)
    cdef int64_t[::1] partner = partner_arr
    cdef cnp.uint8_t[::1] consumed = consumed_arr

    cdef Py_ssize_t ai, lo, hi, mid, pos, j, oi
    cdef Py_ssize_t left, right, best, cl, cr_

    with nogil:
        for ai in range(C):
            if consumed[ai]:
                continue
            # lower_bound: first pos with cres[order[pos]] >= cinv[ai]
            lo = 0
            hi = C
            while lo < hi:
                mid = (lo + hi) >> 1
                if row_ge_key(cres, order[mid], cinv, ai, r):
                    hi = mid
                else:
                    lo = mid + 1
            pos = lo

            left = -1
            j = pos - 1
            while j >= 0:
                oi = order[j]
                if oi != ai and consumed[oi] == 0:
                    left = oi
                    break
                j -= 1

            right = -1
            j = pos
            while j < C:
                oi = order[j]
                if oi != ai and consumed[oi] == 0:
                    right = oi
                    break
                j += 1

            if left == -1:
                best = right
            elif right == -1:
                best = left
            else:
                cl = cpl_row_key(cres, left, cinv, ai, r)
                cr_ = cpl_row_key(cres, right, cinv, ai, r)
                best = right if cr_ > cl else left

            if best != -1 and cpl_row_key(cres, best, cinv, ai, r) > m:
                consumed[ai] = 1
                consumed[best] = 1
                partner[ai] = best

    return partner_arr, consumed_arr.view(np.bool_)
