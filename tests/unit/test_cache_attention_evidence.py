"""Prevent final-only or model-only checks from standing in for stage evidence."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import copy, sys, unittest
from pathlib import Path

ROOT = repository_root(__file__)
sys.path[:0] = [str(ROOT), str(ROOT / "lib")]
from cache_attention_fixtures import check, original
from run_profiles import numerical_summary
from cache_attention_debug import inspect


class CacheEvidence(unittest.TestCase):
    def sample(self):
        batch = dict(
            x=[0.0] * 16,
            query=[0.0] * 16,
            key=[0.0] * 256,
            value=[0.0] * 256,
            wo=[0.0] * 256,
        )
        exact = original(1, 16, 16, batch)
        out = {"result": exact.pop("result")}
        return batch, out, exact

    def test_stage_and_probability_evidence_required(self):
        batch, out, stages = self.sample()
        gate = check(1, 16, 16, batch, out, stages)
        case = dict(
            native_application_checks=[gate],
            device_application_checks=[gate],
            audit=dict(passed=True),
        )
        self.assertTrue(numerical_summary(case)["fixed_accuracy_passed"])
        for broken in ["final_only", "missing_probability", "mass"]:
            bad = copy.deepcopy(case)
            if broken == "final_only":
                bad["device_application_checks"] = [check(1, 16, 16, batch, out)]
            if broken == "missing_probability":
                bad["device_application_checks"][0]["stages"].pop("probability")
            if broken == "mass":
                bad["device_application_checks"][0]["max_row_mass_error"] = 0.02
            with self.assertRaises(ValueError, msg=broken):
                numerical_summary(bad)

    def test_debugger_does_not_invent_absent_intermediates(self):
        from frontend import parse
        from ir import verify
        from mesh_cache_attention import plan

        s = plan(
            verify(
                parse(ROOT / "benchmarks/inference/waferllm/cache_attention_5x256x512_8x8/hls.cpp"),
                8,
                2,
            )
        )
        view = inspect(s, None, "p7_7", 6, 10)
        self.assertFalse(view["available"])
        self.assertIsNone(view["values"])
        s["instrumentation"] = "counters"
        fake = {
            "diagnostics": [
                {"progress": [[[1] * 8] * 8] * 8, "queues": [[[255, 255]] * 8] * 8}
            ]
        }
        view = inspect(s, fake, "p7_7", 0, 9)
        self.assertTrue(view["available"])
        self.assertFalse(view["observed"])
        self.assertIsNone(view["raw_words"])


if __name__ == "__main__":
    unittest.main()
