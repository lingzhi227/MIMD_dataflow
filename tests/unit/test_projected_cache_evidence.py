"""Whole graph evidence cannot be replaced with output-only or invalid norm metrics."""

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
from projected_cache_fixtures import check, original
from run_profiles import numerical_summary
from fixtures import check_application


class ProjectedEvidence(unittest.TestCase):
    def test_all_stages_fixed_finite_metrics_required(self):
        b = dict(
            x=[0.0] * 16,
            gamma=[1.0] * 16,
            wq=[0.0] * 256,
            wk=[0.0] * 256,
            wv=[0.0] * 256,
            cosine=[1.0] * 8,
            sine=[0.0] * 8,
            key=[0.0] * 256,
            value=[0.0] * 256,
            wo=[0.0] * 256,
        )
        observed = original(1, 16, 16, b)
        outputs = dict(
            result=observed.pop("result"),
            new_key=observed.pop("rotated_key"),
            new_value=observed.pop("value_projection"),
        )
        gate = check(1, 16, 16, b, outputs, observed)
        case = dict(
            native_application_checks=[gate],
            device_application_checks=[gate],
            audit=dict(passed=True),
        )
        self.assertTrue(numerical_summary(case)["fixed_accuracy_passed"])
        mutations = [
            lambda v: v["metrics"].pop("query"),
            lambda v: v.update(all_stage_gates=False),
            lambda v: v["limits"].update(relative_l2=0.05),
            lambda v: v.update(max_probability_mass_error=float("nan")),
            lambda v: v.update(max_probability_mass_error=True),
            lambda v: v["metrics"]["score"].update(relative_l2=0.021),
            lambda v: v["metrics"]["score"].update(relative_peak=float("nan")),
            lambda v: v["metrics"]["score"].update(relative_peak=-1),
            lambda v: v["metrics"]["score"].update(relative_peak=True),
        ]
        for mutation in mutations:
            bad = copy.deepcopy(case)
            mutation(bad["device_application_checks"][0])
            with self.assertRaises(ValueError):
                numerical_summary(bad)
        partial = check_application("projected_cache:1:16:16", b, outputs)
        with self.assertRaises(ValueError):
            numerical_summary(dict(native_application_checks=[partial]))
        with self.assertRaises(AssertionError):
            check(1, 16, 16, b, outputs, dict(score=observed["score"]))
