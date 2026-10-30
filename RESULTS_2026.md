# 2026 computations: evidence inventory

## Euler record

The retained `build_floor1300000.log` reports a construction using the modulus
`Λ = 2^9·3^5·5^3·7^2·11^2·13^2·17·19·23·29·31·37·41·43·47·53·59·61·67·71·73`.
The smallest allowed prime factor was 1,300,000. The run wrote
`euler_floor1300000.txt`: 8,667,018 factors, one decimal integer per line. The
separate `verify_floor1300000.log` reports an Euler pseudoprime for the first
90,968 prime bases, ending at 1,172,819. The builder generates its prime pool
with Lucas/Pocklington primality proofs, using the known factorization of
`p−1`. The retained verifier also checked all 8,667,018 output factors with
strong probable-prime tests. A separate rigorous-primality pass over the saved
factor file is not recorded. A small exact sieve separately confirmed that
1,172,819 is the 90,968th prime.

The full-record verification command needs `euler_floor1300000.txt`, the list
of 8,667,018 factors. The source repository includes its compressed form at
[`data/euler_floor1300000.txt.gz`](data/euler_floor1300000.txt.gz). Decompress
it before running the verifier. The sizes and SHA-256 hashes below identify
both forms and the included logs.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `euler_floor1300000.txt` | 173,559,103 | `32ec92285a2a57299f73cbdc01f80daf758766b7d9a9def43e5c785b3d1a9c1c` |
| `data/euler_floor1300000.txt.gz` | 71,247,144 | `0511d4480de51232f010017e9ca0b7011841bff4ccf338a5fef211e2aac2bab9` |
| `evidence/build_floor1300000.log` | 8,251 | `8c290a7ef74f930ad0e64b7c589b250bcd4615b2dfb13d742f9c56d57958c21b` |
| `evidence/verify_floor1300000.log` | 641 | `5657d886d414d93bc7b3f85d9b4ef1a3798f1dc5fba9582f324d370f086e97ae` |

The construction-independent fast verifier can check a supplied factor file:

```sh
gzip -cd data/euler_floor1300000.txt.gz > euler_floor1300000.txt
python3 tools/verify_euler.py \
  --lambda '2^9*3^5*5^3*7^2*11^2*13^2*17*19*23*29*31*37*41*43*47*53*59*61*67*71*73' \
  euler_floor1300000.txt --max-base 90968 -j 1
```

Add `--full-primality` to have the standalone verifier repeat rigorous
Lucas/Pocklington proofs over the saved factor file.
`humanverify_euler.py` is a separate checker, but no full-record run of that
checker is included here.

The compressed factor list was made with `gzip -n -9`.
Its gzip integrity check passed, and streaming decompression had the raw file's
SHA-256 shown above. Check the compressed hash before unpacking and the raw
hash afterward.

```sh
sha256sum data/euler_floor1300000.txt.gz
gzip -cd data/euler_floor1300000.txt.gz | sha256sum
```

## Carmichael factor-count coverage

The retained `aghs_79_83.log` reports 340,210,290 generated primes and 1,529,840
base Carmichael numbers. It reports every factor count from 32 through
340,210,163, along with isolated counts 30, 340,210,165, and 340,210,195.
The published Alford–Grantham–Hayman–Shallue result covers every count from 3
through 19,565,220. These intervals overlap. Together they give the reported
range 3 through 340,210,163, using the 2014 result and this run log.

The log uses
`Λ = 2^12·3^7·5^4·7^3·11^2·13^2·17^2·19·23·29·31·37·41·43·47·53·59·61·67·71·73·79·83`.
Its SHA-256 is
`82ca82ff7d75ee52cebe1d49459ce0b085e6f02d75ce1942fd3d6629665c550a`
for the byte-identical `evidence/aghs_79_83.log` copy.
No factored base-number witness for that run is included here. The run log by
itself does not expose each base number's prime factors and congruence for
independent reconstruction.
