"""Which factor counts ``k`` are achievable.

The base numbers are pairwise coprime and each is ``== 1 (mod Lambda)``, so the product of
*any* subset is again ``== 1 (mod Lambda)`` and Carmichael (when it has >= 3 prime factors).
A product's number of prime factors is the sum of the factor counts of the chosen base
numbers.  Hence the set of achievable ``k`` is exactly the set of subset sums of the
multiset of base factor counts.

We compute it as a bitset using Python's arbitrary-precision integers: bit ``k`` of
``reach`` is set iff some subset sums to ``k``.  This is the modern one-liner version of
the vector ``v`` shifted-and-added in the paper's prose.
"""

from __future__ import annotations

from collections import Counter

import numpy as np


def reachable_counts(factor_counts: list[int]) -> int:
    """Return a bitset (Python int) whose set bits are the achievable subset sums.

    Bit 0 (the empty product) is always set; genuine Carmichael numbers need ``k >= 3``.

    Repeated factor-counts are handled by **binary splitting** (bounded-knapsack style):
    ``v`` copies of size ``f`` are realized in ``O(log v)`` shift-ors (groups of
    ``1, 2, 4, ...`` copies) instead of ``v``.  The result is identical -- subset-sum
    reachability is order-independent -- but the op count drops from ``sum(v_f)`` to
    ``sum(ceil(log2(v_f+1)))``, which is tens-to-hundreds x fewer when many base numbers
    share a factor count (the usual case at scale).
    """
    reach = 1  # only k = 0 reachable initially (empty product)
    for f, v in Counter(factor_counts).items():
        g = 1
        while v > 0:
            take = g if g < v else v      # groups 1, 2, 4, ..., remainder
            reach |= reach << (f * take)
            v -= take
            g <<= 1
    return reach


def achievable_ks(reach: int, lo: int = 3) -> np.ndarray:
    """Positions ``>= lo`` of the set bits of ``reach``, as a sorted ``int64`` array.

    Single ``to_bytes`` + ``unpackbits`` pass -- ``O(bit length of reach)``.  (The obvious
    ``[k for k in range(3, top+1) if (reach >> k) & 1]`` is ``O(top^2)``: it reshifts the
    whole multi-megabit integer once per bit, which is hours when ``top`` is ~10^8.)
    """
    nbits = reach.bit_length()
    if nbits <= lo:
        return np.empty(0, dtype=np.int64)
    buf = reach.to_bytes((nbits + 7) // 8, "little")
    bits = np.unpackbits(np.frombuffer(buf, dtype=np.uint8), bitorder="little")
    pos = np.flatnonzero(bits).astype(np.int64)
    return pos[pos >= lo]


def reachable_set(factor_counts: list[int]) -> list[int]:
    """The sorted list of achievable ``k >= 3`` (i.e. those that are Carmichael)."""
    return achievable_ks(reachable_counts(factor_counts)).tolist()


def format_ranges(ks) -> str:
    """Collapse a sorted sequence of ints (list or ``np.ndarray``) into ``"a-b c d-e"``
    range notation.  Vectorized: cost is ``O(len)`` for the gap scan, not per-element."""
    arr = np.asarray(ks, dtype=np.int64)
    if arr.size == 0:
        return "(none)"
    brk = np.flatnonzero(np.diff(arr) > 1)          # last index of each contiguous run
    starts = np.concatenate(([0], brk + 1))
    ends = np.concatenate((brk, [arr.size - 1]))
    parts = []
    for s, e in zip(starts.tolist(), ends.tolist()):
        a, b = int(arr[s]), int(arr[e])
        parts.append(f"{a}-{b}" if a != b else f"{a}")
    return " ".join(parts)


def base_histogram(factor_counts: list[int]) -> dict[int, int]:
    """Map factor-count -> number of base Carmichael numbers having it."""
    hist: dict[int, int] = {}
    for f in factor_counts:
        hist[f] = hist.get(f, 0) + 1
    return hist
