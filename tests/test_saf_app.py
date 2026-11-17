import sys
from importlib.metadata import metadata

import pytest

from diffpy.saf import saf_app


def test_main_help_shows_package_description(monkeypatch, capsys):
    # C1: The help flag is given.
    # Expected: The help text shows the description in pyproject.toml.
    monkeypatch.setattr(sys, "argv", ["diffpy-saf", "--help"])
    with pytest.raises(SystemExit):
        saf_app.main()
    actual_help = " ".join(capsys.readouterr().out.split())
    expected_description = metadata("diffpy.saf")["Summary"]
    assert expected_description in actual_help
