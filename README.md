# Euler and Carmichael computations

This package constructs Carmichael numbers with specified numbers of prime
factors and Euler pseudoprimes to many prime bases. It includes output checkers
and factored examples.

## Copyright and license

© Institute for Defense Analyses (IDA), 2026. The code was developed by the IDA
Center for Computing Sciences operating as a Federally Funded Research and
Development Center using U.S. Government funds.

The code is provided under the [MIT License](LICENSE).

## 2026 results and evidence

The retained Carmichael run log reports every prime-factor count from 32 through
340,210,163. Combined with the published 2014 coverage from 3 through
19,565,220, this gives the stated range from 3 through 340,210,163.

The retained Euler verifier reports an Euler pseudoprime to the first 90,968
prime bases, through 1,172,819, with 8,667,018 factors. The builder proves
members of its prime pool with Lucas/Pocklington tests; the retained verifier
also screened every output factor with strong probable-prime tests. See
[RESULTS_2026.md](RESULTS_2026.md) for run reports, hashes, and commands.

The compressed factor list is included at
[data/euler_floor1300000.txt.gz](data/euler_floor1300000.txt.gz).
For the checkers' mathematical basis, see [VERIFICATION.md](VERIFICATION.md).

The construction follows Algorithm 1 of W. R. Alford, Jon Grantham,
Steven Hayman, and Andrew Shallue, “Constructing Carmichael numbers through
improved subset-product algorithms,” *Mathematics of Computation* **83**
(2014), 899–915 ([arXiv:1203.6664](https://arxiv.org/abs/1203.6664)).

## What the algorithm does

The Erdős construction builds a Carmichael number from a modulus

```
Λ = ∏ qᵢ^hᵢ        and the prime set   P = { p prime : p−1 | Λ,  p ∤ Λ }.
```

Any subset `S ⊆ P` with at least three primes whose product is `≡ 1 (mod Λ)` is
squarefree and satisfies Korselt's criterion. Its product is therefore a
Carmichael number. Algorithm 1 finds *many* such subsets at once, producing a
collection of pairwise-coprime **base** Carmichael numbers; the product of any
sub-collection is again Carmichael. Because the base numbers have a spread of
factor counts, the achievable totals `k` (subset sums of the factor counts)
cover a wide range. The 2014 Alford–Grantham–Hayman–Shallue paper reports
every `k` from 3 to 19,565,220.

### Generating the prime set

The program enumerates divisors `d` of `Λ` and retains `p = d + 1` when `p` is
prime and `p ∤ Λ`. The factorization of `d = p − 1` is known from `Λ`, so the
prime-generation routine uses a Lucas/Pocklington test to prove each retained
`p` prime. Pseudoprimes occur among the candidates, making this check necessary.

For paper-scale `Λ`, building `P` can be costly: the divisor sweep has hundreds
of millions of candidates, each needing a primality proof. Two levers make
prime generation tractable:

- **Parallelism.** The divisor recursion is partitioned into independent
  *prefix-branch subtrees* (fix the exponents of the leading primes, enumerate
  the rest), one process per subtree via `ProcessPoolExecutor`. Each worker
  proves its own shard; the parent concatenates and sorts into a canonical,
  scheduling-independent order. Controlled by `--jobs` (default
  `min(cpu_count, 72)`); `--jobs 1` forces the serial path, which is also used
  automatically below ~200k divisors.
- **gmpy2.** When `gmpy2` is installed, the modular exponentiation in the proof
  uses GMP (`HAVE_GMPY2` switch, mirroring the Cython one). On the ~100–130-bit
  moduli here this is ≈9× faster than CPython `pow`. The package also runs
  without it. The two backends are checked for identical verdicts in
  `tests/test_parallel.py`.

Measurements from a 72-core host (with Lucas/Pocklington primality proofs):

| `Λ` preset | divisors | \|P\| | wall time | peak RSS |
|---|---|---|---|---|
| `lambda-1998` | 10.6 M | 1,371,578 | **6.4 s** | 0.7 GB |
| `record-alg1` | 230 M | 24,803,689 | **3.9 min** | 14.3 GB |

`record-alg1` produces `|P| = 24,803,689 > 19,565,220`. This meets a cardinality
requirement for that range but does not establish contiguous coverage;
the factor counts of the base numbers matter.

### The "push down"

Write `Λ` as a product of prime powers `Q₀, …, Qᵣ₋₁`. Represent each element of
`(Z/ΛZ)*` by its residues modulo these prime powers, ordered by descending value
of `Qᵢ`. An element is "done through a prefix of length `L`" when its first `L`
residues are all `1`. Sweeping the columns `m = 0,1,…,r−1` (largest prime power
first), at each stage we:

1. **fix the product** — discard at most one element so the product of all
   survivors is `≡ 1 (mod Qₘ)` (the "throw away one number" step);
2. **push down** — greedily pair elements so each product clears column `m`.
   Keeping the working set sorted lexicographically by residue vector, the best
   partner for `a` is an immediate neighbor of `a⁻¹` in that order, found by
   binary search (this is the efficient match described in the 2014 paper's
   *Construction* section);
3. **sweep the chaff** — everything unpaired multiplies to a single element that
   is `≡ 1` on columns `0..m`; carry it forward.

After the last stage every survivor is `≡ 1 (mod Λ)`. Survivors with at least
three distinct prime factors are base Carmichael numbers.

## Layout

| file | role |
|------|------|
| `pdc/lam.py` | stores `Λ` and its ordered prime-power columns |
| `pdc/primes.py` | builds `P` and proves each member prime |
| `pdc/pushdown.py` | Algorithm 1 itself |
| `pdc/verify.py` | Korselt verification of every output |
| `pdc/coverage.py` | subset-sum DP over factor counts (bitset) |
| `pdc/cli.py` | driver and report |

## Usage

```bash
cd euler-carmichael-computations

# Run the default demo and report its factor counts:
python3 -m pdc

# Choose your own Λ:
python3 -m pdc --lambda "2^8*3^4*5^2*7^2*11*13*17*19*23*29*31"

# Use a named preset and control parallelism (default: min(cpu_count, 72)):
python3 -m pdc --preset lambda-1998 --jobs 64    # ~1.37M primes
# This larger preset used 25.9 GB peak RSS in the recorded run:
python3 -m pdc --preset record-alg1

# Write the base Carmichaels in factored form:
python3 -m pdc --write-carmichaels mybase.txt

# Verify the included 1998 factored output:
python3 -m pdc --verify-file tests/fixtures/carmichaels.golden
```

Selected lines from the default run (~1 s):

```
Lambda = 2^7 * 3^4 * 7^2 * 5^2 * 23 * 19 * 17 * 13 * 11 = 13492656777600
|P| = 2671 primes generated and proven
Found 16 base numbers (16 Carmichael, 0 with <3 factors).
All base numbers verified == 1 (mod Lambda); all >=3-factor numbers pass Korselt.
Carmichael numbers exist with k prime factors for k in:
  13-14 21-22 27-28 30-31 33-36 41-944 ... 1700-2603 ... 2644
```

## Tests

The test suite requires the `test` optional dependencies (`pytest` and `gmpy2`);
these are not needed for the pure-Python construction and fast verifier
fallback.

```bash
python3 -m pytest
```

`test_golden.py` re-verifies all 70 Carmichael numbers from the original 1998
run (`tests/fixtures/carmichaels.golden`); `test_pushdown.py` checks every
constructed base number is `≡ 1 (mod Λ)` and passes Korselt, and that the
factor-count coverage matches a brute-force subset-sum.

## "Core + fast path"

The implementation uses NumPy arrays. Residue vectors live in an `int64` matrix
and per-modulus inverse tables are precomputed, so the arithmetic is vectorized.
The element representation — residue vector + a *history* of constituent prime
indices, never the giant product — is what makes scaling possible: exact
integers are reconstructed only at verification time.

### The Cython hot loop

The hot spot of Algorithm 1 is the per-stage neighbor match (lexicographic
binary search + neighbor scan + common-prefix length, for every candidate). It
lives in `pdc/_match.pyx` as an integer-only `nogil` kernel, `match_stage`, that
works directly on the `int64` residue matrix. `pushdown._match_stage` dispatches
to it when the extension is built and otherwise falls back to `_match_stage_py`,
a line-for-line pure-Python reference; the two are checked for **identical**
pairings in `tests/test_match_equivalence.py` (200 randomized stages + a
full-run comparison).

The extension is optional when running from the checkout. Installing with
`pip install -e .` builds the extension and requires Cython, NumPy, and a C
compiler. To build it in place:

```bash
python3 setup.py build_ext --inplace
```

Measured on `Λ = 2^8·3^4·5^3·7^2·11^2·13·…·31` (`|P| = 19,048`; first stage has
16,933 candidates):

| | matcher kernel (one stage) | full push-down (all stages) |
|---|---|---|
| pure Python | 241 ms | 668 ms |
| Cython | 1.9 ms | 117 ms |
| speedup | **126×** | **5.7×** |

The kernel is two orders of magnitude faster. The rest of each stage is also
vectorized: merged pairs are realized as a single `int64` block product (a
pairwise residue product is `< Q_k²`, so it stays exact in `int64`; an extreme
`Q` falls back to exact Python objects), and the chaff is reduced with a
pairwise tree (`_modprod_rows`).

### Push-down and verification at scale

The pipeline handles large prime sets with these data structures:

- **Push-down representation.** The global-product fix-up and discard
  tie-break use vectorized operations (`_modprod_rows` + a NumPy common-prefix
  `argmax`). Each working element holds a node ID in a merge forest. Pairs and
  chaff append forest nodes. The forest is flattened to prime-index lists
  once at the end, avoiding O(N·r) list-copying.
- **Verification without the integer.** A base number can have tens of millions
  of prime factors, so reconstructing `n = ∏p` (hundreds of millions of digits,
  O(K²)) is hopeless. `verify_base_numbers` instead uses the **Korselt-via-Λ**
  argument: every `p∈P` has `(p−1) | Λ` and the push-down forces
  `n ≡ 1 (mod Λ)`, so `(p−1) | Λ | (n−1)` for free. It checks `(p−1)|Λ` once
  over `P`, then per base confirms distinct factors, `≥3` of them, and
  `n ≡ 1 (mod Λ)` recomputed as a per-column residue product — all O(N·r)
  small-integer work. (`is_carmichael` reconstructs `n` exactly for the
  small golden-file path.)

Recorded wall times on a 72-core host (parallel generation, push-down,
verification, and coverage):

| Λ preset | \|P\| | push-down | verify | total | peak RSS |
|---|---|---|---|---|---|
| `lambda-1998` | 1,371,578 | 10.1 s | 1.7 s | **19.6 s** | 1.3 GB |
| `record-alg1` | 24,803,689 | 534 s | 37 s | **14 min** | 25.9 GB |
| `record-flat` | 21,823,970 | 575 s | 34 s | **14 min** | 23.5 GB |

The `record-flat` run produced about 10⁵ verified base Carmichael numbers.
The largest had millions of prime factors.

### Factor-count coverage from one modulus

The 2014 paper reports Carmichael numbers with *every* `k` prime factors
across a contiguous range up to 19,565,220. Whether the achievable `k`
(subset sums of the base factor counts) are *contiguous* depends on the
**shape** of Λ:

- A Λ with few primes / high exponents (`record-alg1`) makes the push-down
  produce a few *dominant* base numbers, leaving large gaps — `|P| > 19.5M` is
  **not** sufficient.
- A **flat** Λ — many distinct primes, low exponents (`record-flat`) — makes it
  produce *many* base numbers with a dense size spectrum and a small dominant
  base. The subset-sum bitset computation reports which factor counts are
  attained; its reported interval is checked for gaps.

`record-flat` gives **every `k` from 22 to 21,823,910** from a single modulus.
The factor counts `3…21` come from known small Carmichael numbers. A heuristic
count suggests that small-factor outputs from this large modulus are unlikely;
it does not rule them out.

The retained `aghs_79_83` run reports every `k` from 32 through 340,210,163.
Its interval overlaps the 2014 result's range from 3 through 19,565,220,
giving the stated coverage from 3 through 340,210,163. Its run log is included
under `evidence/`.

## Euler pseudoprimes to many prime bases (`pdc/euler.py`)

A downstream application of the same coprime-Carmichael substrate. The
Solovay–Strassen test calls `n` prime for base `a` when
`a^{(n-1)/2} ≡ (a/n) (mod n)` (a Jacobi symbol); a composite that passes is an
**Euler pseudoprime** to base `a`. Improving on *"Ours goes to 211: Euler
Pseudoprimes to 47 Prime Bases"* (Alcantarilla Sánchez–Cottaar–Lange–de Weger),
which multiplied Carmichaels pairwise, we **solve** for a good product by linear
algebra over `F₂`.

**The idea.** Write `λ(n)` for Carmichael's function. A Carmichael number `n` is
in **class A** when its index `i(n)=(n-1)/λ(n)` is even. Equivalently,
`v₂(n-1) > maxₚ v₂(p-1)`. For every `a` coprime to such an `n`,
`a^{(n-1)/2} ≡ 1 (mod n)`, so

> `n` (class A) is an Euler pseudoprime to base `a`
> ⇔ the Jacobi symbol `(a/n) = +1`.

Generate `P` from a flat `Λ` with two filters
(`generate_prime_set(..., class_a=True, min_prime=…)`): dropping primes with
`v₂(p-1)=v₂(Λ)` forces every output Carmichael number into class A, and the
floor keeps `n` coprime to every test base. Then `(a/n) = ∏ⱼ (a/Nⱼ)`, so the bit
`[(a/n)=-1]` is **linear over `F₂`** in the choice of which base Carmichaels
`Nⱼ` to multiply. A nonzero vector in the kernel of `M[q, j] = [(q/Nⱼ)=-1]`
(rows = bases `q ∤ Λ`; bases `q | Λ` are automatic) is a subset product that is
simultaneously an Euler pseudoprime to **all** target bases. With `m` coprime
base Carmichaels ≫ #bases, a solution is *guaranteed*, not hoped for. Euler
liars form a subgroup, so surviving the first `t` primes ⇒ surviving every base
below `p_{t+1}`.

```bash
# Build (maximize) and write the prime factors of n:
python3 tools/build_euler.py --preset record-flat --maximize \
        --floor 224738 --want-sparse --out record_flat_euler_bases.txt
# Verify independently of the construction code:
python3 tools/verify_euler.py \
        --lambda "2^9*3^5*5^3*7^2*11^2*13^2*17*19*23*29*31*37*41*43*47*53*59*61*67*71*73" \
        record_flat_euler_bases.txt --max-base 20000
```

The `record-flat` run used 19,555,732 primes and 90,612 coprime base
Carmichael numbers. It produced an **Euler pseudoprime to the first 20,000
prime bases**, hence to every integer base below 224,743. It is the product of
9,708 base Carmichaels: **2,406,002 prime factors, ~1.5×10⁸ bits**, written in
factored form to `record_flat_euler_bases.txt`. The reported matrix rank was
19,979 against 90,612 columns.
`tools/verify_euler.py` checks factors by strong probable-prime tests by
default; `--full-primality` requests a rigorous proof for each factor. It also
checks that `n` is a class-A Carmichael, and it is an Euler liar for all 20,000
bases. Small test cases compare the reduced checker with the literal
`a^{(n-1)/2} mod n` Solovay–Strassen test.
