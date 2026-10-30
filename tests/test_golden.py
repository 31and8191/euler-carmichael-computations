"""Validate against the original 1998 output (tests/fixtures/carmichaels.golden)."""

from pathlib import Path

import pytest

from pdc.verify import read_factored_carmichaels, is_carmichael

GOLDEN = Path(__file__).resolve().parent / "fixtures" / "carmichaels.golden"


@pytest.mark.skipif(not GOLDEN.exists(), reason="carmichaels.golden not present")
def test_all_golden_numbers_are_carmichael():
    groups = read_factored_carmichaels(str(GOLDEN))
    assert len(groups) == 70  # matches out.golden: "Found 70 Carmichaels"
    for g in groups:
        chk = is_carmichael(g)
        assert chk.ok, f"{g[:3]}... failed: {chk.reason}"
