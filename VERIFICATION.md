# How the output checkers work

The checkers in `tools/` read factored output without importing the
construction package `pdc`. An Euler input lists one proposed prime factor per
line; its checkers ignore blank lines and `#` comments. A Carmichael input
lists one factor per line, with a blank line between base numbers. Its checker
expects only factor lines and blank separators. In either case, let `n` be the
product of one list of factors and let `Λ` be the supplied modulus.

## Carmichael condition

For each group with at least three factors, the Carmichael checker confirms
distinct factors, `(p−1) | Λ` for each factor `p`, and `n ≡ 1 (mod Λ)`. It
obtains the last residue by multiplying `p mod Λ`; it need not form the large
integer `n`. If every factor is prime, then `(p−1) | Λ | (n−1)` for each `p`.
The product is therefore Carmichael by Korselt's criterion. The checker reports
groups with fewer than three factors separately as building blocks. It does not
check that separate groups have disjoint factors, which is needed when their
factor counts are combined by subset sums.

The builder proves members of its prime pool using the known factorization of
`p−1` and a Lucas/Pocklington test. The standalone checkers screen factors
again. `tools/independent_verify.py` and `tools/verify_euler.py` use strong
probable-prime tests by default; `--full-primality` requests rigorous proofs
for each factor. These scripts validate that the listed factors of `Λ` are
distinct primes before using them for rigorous proofs.
`tools/humanverify_euler.py` uses `gmpy2.is_prime`, which is a
probable-prime test for large factors. Without a rigorous factor check, the
standalone Carmichael conclusion depends on the listed factors being prime.

## Euler condition

For a base `a` coprime to `n`, the Euler-pseudoprime condition is

```text
a^((n−1)/2) ≡ (a/n) (mod n),
```

where `(a/n)` is the Jacobi symbol. Once the factors are established as
distinct primes, the Chinese remainder theorem reduces this to one
congruence modulo each factor `p`. The Jacobi symbol is the product of the
Legendre symbols `(a/p)`.

To compute the exponent without forming `n`, the human-readable checker first
multiplies the factors modulo `2Λ`, obtaining `R = n mod 2Λ`. Since
`(p−1) | Λ`, its exponent modulo `p−1` is

```text
eₚ = ((R−1) mod 2(p−1)) // 2.
```

It checks `a^eₚ mod p` against the Jacobi symbol for every `p`. The fast checker
uses the same congruence with a shortcut: because `(p−1) | (n−1)`, Euler's
criterion makes the local value `1` when `(n−1)/(p−1)` is even and `(a/p)`
when it is odd. Both methods avoid exponentiation modulo the full `n`.

For small inputs, both Euler checkers can also evaluate the defining
congruence modulo `n` directly. The included
`tests/fixtures/euler_golden.txt` exercises this comparison. If the tested
prime bases all pass and every smaller integer is coprime to `n`, the subgroup
property of Euler-liar bases extends the result to every integer base below
the next prime.

## Checker scripts

| Script | Input | Role |
|---|---|---|
| `tools/independent_verify.py` | factored base numbers | Carmichael check |
| `tools/verify_euler.py` | one factor list | fast Euler check |
| `tools/humanverify_euler.py` | factors | loop-based Euler check |

The included full-record Euler log is from `tools/verify_euler.py`, using its
default strong probable-prime factor screen. See
[RESULTS_2026.md](RESULTS_2026.md) for the input file, commands, hashes, and
the exact scope of the retained Euler and Carmichael runs.
