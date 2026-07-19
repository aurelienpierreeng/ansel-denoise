"""pack_contribution.py handle validation (placeholder + charset guards)."""

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pack_contribution.py"


def _run(handle, tmp):
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp), "--handle", handle, "--yes"],
        capture_output=True, text=True)


def test_placeholder_handles_refused(tmp_path):
    for ph in ["your-github-name", "your-name", "YOUR-GITHUB-NAME", "handle", "username"]:
        r = _run(ph, tmp_path)
        assert r.returncode != 0, f"{ph} should be refused"
        assert "placeholder" in r.stderr.lower(), r.stderr


def test_bad_charset_refused(tmp_path):
    r = _run("Alain Ahs", tmp_path)  # space + uppercase
    assert r.returncode != 0 and "lowercase letters" in r.stderr


def test_real_handle_passes_the_guard(tmp_path):
    # a valid handle clears the handle checks and proceeds to shard validation,
    # which fails on the empty dir — proving the guard itself let it through
    r = _run("alain-ahs", tmp_path)
    assert r.returncode != 0
    assert "placeholder" not in r.stderr.lower()
    assert "no shards" in r.stderr.lower() or "no shards" in r.stdout.lower()
