"""The bounded class-A comparison must keep 2Q and 4Q distinct."""

import pytest

from tools.compare_class_a_methods import _method_kwargs


def test_comparison_method_interface_distinguishes_all_three_paths():
    assert _method_kwargs("filter") == {"class_a": True}
    assert _method_kwargs("2Q") == {"two_adic_multiplier": 2}
    assert _method_kwargs("4Q") == {"two_adic_multiplier": 4}
    assert _method_kwargs("2Q") != _method_kwargs("4Q")


def test_comparison_method_interface_rejects_unknown_label():
    with pytest.raises(ValueError, match="unknown class-A method"):
        _method_kwargs("8Q")
