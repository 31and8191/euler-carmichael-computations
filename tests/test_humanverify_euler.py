"""The human-auditable verifier (tools/humanverify_euler.py) must accept the committed golden
Euler pseudoprime, agree with the literal Solovay-Strassen definition on a small case, and
reject a non-Carmichael input."""

import sys
from pathlib import Path

import gmpy2  # noqa: F401  (ensures the optional dep is present, as the tool requires it)

from pdc.lam import preset
from pdc.primes import generate_prime_set
from pdc.pushdown import push_down
from pdc.euler import construct_euler_pseudoprime, first_primes

_TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(_TOOLS))
import humanverify_euler as hv  # noqa: E402

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "euler_golden.txt"
_GOLDEN_LAMBDA = "2^8*3^5*5^3*7^2*11^2*13^2*17*19*23*29*31*37"


def test_golden_fixture_verifies():
    """The committed golden example must verify as an Euler pseudoprime to the first 60 primes."""
    assert _FIXTURE.exists(), "golden fixture missing"
    assert hv.main(["--lambda", _GOLDEN_LAMBDA, str(_FIXTURE), "--max-base", "60"]) == 0


def test_literal_definition_anchor_small(tmp_path):
    """On a small n the transparent per-prime method must agree with the literal
    a**((n-1)//2) mod n == (a/n) check (the tool returns nonzero on any disagreement)."""
    lam = preset("demo")
    P = generate_prime_set(lam, min_prime=first_primes(20)[-1] + 1, class_a=True)
    e = construct_euler_pseudoprime(push_down(P), 20, want_sparse=True)
    f = tmp_path / "demo20.txt"
    f.write_text("".join(f"{p}\n" for p in e.factor_values))
    n_bits = sum(p.bit_length() for p in e.factor_values)
    # default --modexp-limit (40000) exceeds this n, so the literal cross-check runs
    assert n_bits < 40_000
    assert hv.main(["--lambda", str(lam), str(f), "--max-base", "20"]) == 0


def test_rejects_non_carmichael():
    """A wrong Lambda (so (p-1) does not divide it) must be rejected, not silently passed."""
    assert hv.main(["--lambda", "2^8*3^5", str(_FIXTURE), "--max-base", "10"]) == 1
