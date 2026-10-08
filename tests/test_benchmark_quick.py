"""End-to-end benchmark in quick mode on CPU, plus the 'one number' guards."""

import json
import re

import pytest

from benchmark import protocol as P
from benchmark import report, run

QUICK = ["--quick", "--device", "cpu", "--random-weights"]


@pytest.fixture(scope="module")
def quick_results(tmp_path_factory):
    out = tmp_path_factory.mktemp("bench") / "b.json"
    return run.main(QUICK + ["--out", str(out)]), out


def test_quick_run_writes_json_and_markdown(quick_results):
    res, out = quick_results
    on_disk = json.loads(out.read_text())
    assert on_disk["protocol"]["version"] == P.PROTOCOL_VERSION
    assert on_disk["random_weights"] and on_disk["quick"]
    assert out.with_suffix(".md").read_text().startswith("# Benchmark results")
    scales = {e["scale"] for e in res["quality"]}
    assert scales == set(P.SCALES)
    methods = {e["method"] for e in res["quality"]}
    assert {"bicubic", "imdn_thermal"} <= methods
    assert {e["scale"] for e in res["speed"]} == set(P.SCALES)
    for e in res["speed"]:
        assert e["batch"] == 1 and e["model"]["median_ms"] > 0
        assert e["model"]["fps"] == pytest.approx(1000 / e["model"]["median_ms"])


def test_quality_is_deterministic(quick_results, tmp_path):
    res, _ = quick_results
    again = run.main(QUICK + ["--no-speed", "--out", str(tmp_path / "c.json")])

    def key(r):
        return {
            (e["set"], e["scale"], e["degradation"], e["method"], e["precision"]): e["per_image"]
            for e in r["quality"]
        }

    assert key(res) == key(again)


def test_replace_blocks():
    res = {
        "quality": [],
        "speed": [],
        "test_sets": {},
        "protocol": {"version": "x"},
        "environment": {"torch": "t", "python": "p"},
        "command": "c",
        "generated_utc": "u",
    }
    text = "a\n<!-- BENCHMARK:SPEED:START -->\nold\n<!-- BENCHMARK:SPEED:END -->\nb\n"
    new = report.replace_blocks(text, res)
    assert "old" not in new and "Speed not measured" in new
    assert new.startswith("a\n") and new.endswith("b\n")


def test_docs_are_in_sync_with_results():
    """README / portfolio / results.md must be regenerated from results/benchmark.json."""
    if not report.RESULTS.exists():
        pytest.skip("no committed results")
    assert report.main(["--check"]) == 0


def test_readme_has_no_unsupported_superlatives():
    text = (P.REPO_ROOT / "README.md").read_text()
    for pat in (
        r"\bSOTA\b",
        r"state[- ]of[- ]the[- ]art",
        r"15x faster",
        r"\bFirst\b",
        r"[Bb]reakthrough",
        r"[Nn]ovel IMDN",
    ):
        hits = [m.group(0) for m in re.finditer(pat, text)]
        # allowed only inside the 'what changed' list that quotes the removed claims
        quoted = re.findall(r"~~[^~]*~~", text)
        assert all(any(h in q for q in quoted) for h in hits), f"{pat!r} outside a quote"
