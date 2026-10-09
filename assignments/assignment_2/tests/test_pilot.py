"""Pilot commands must not overwrite previous runs."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_pilot


@pytest.mark.parametrize("name", ["", ".", "..", "../previous", "/tmp/previous"])
def test_rejects_paths(monkeypatch, name):
    monkeypatch.setattr(sys, "argv", ["run_pilot.py", name])
    with pytest.raises(SystemExit) as error:
        run_pilot.main()
    assert error.value.code == 2


def test_rejects_zero_workers(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_pilot.py", "unused", "--workers", "0"])
    with pytest.raises(SystemExit) as error:
        run_pilot.main()
    assert error.value.code == 2


def test_preserves_existing_directory(monkeypatch, tmp_path):
    marker = tmp_path / "keep.txt"
    marker.write_text("previous results")
    monkeypatch.setattr(sys, "argv", ["run_pilot.py", "existing"])
    monkeypatch.setattr(run_pilot, "outputs", lambda _: {
        "results_dir": tmp_path / "results", "figures_dir": tmp_path / "figures",
    })
    with pytest.raises(FileExistsError):
        run_pilot.main()
    assert marker.read_text() == "previous results"
