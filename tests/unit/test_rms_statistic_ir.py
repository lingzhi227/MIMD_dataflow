
from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)

import copy
import sys
import unittest
from pathlib import Path

ROOT = repository_root(__file__)
sys.path.insert(0, str(ROOT / "lib"))
from frontend import parse, Error
from rms_statistic_ir import StatisticType, verify_connection, mean_boundary


class StatisticContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = parse(ROOT / "experiments/axis_mean/hls-statistic.cpp")

    def test_frontend_boundary(self):
        x, g, n, _ = copy.deepcopy(self.module["nodes"])
        r = mean_boundary(n, x, g, derived_input_bound=34.5625)
        self.assertEqual(r["statistic_type"]["divisor"], 256)
        self.assertEqual(r["geometry"]["participants"], 16)
        self.assertLess(r["normalized_range"]["l1_bound"], 259)

    def test_no_statistic_reinterpretation(self):
        good = StatisticType("mean", 256, 256)
        verify_connection(good, good)
        for bad in (
            StatisticType("sum", 256, 1),
            StatisticType("mean", 128, 128),
            StatisticType("mean", 256, 128),
            StatisticType("mean", 256, 256, result_dtype="f32"),
            StatisticType("mean", True, True),
        ):
            with self.subTest(bad=bad), self.assertRaises(Error):
                verify_connection(good, bad)

    def test_graph_edges_and_policy(self):
        for mutation in (
            "gamma_edge",
            "gamma_shape",
            "gamma_dtype",
            "gamma_bound",
            "axis",
            "unknown",
            "producer",
        ):
            x, g, n, _ = copy.deepcopy(self.module["nodes"])
            if mutation == "gamma_edge":
                g["id"] = "other"
            elif mutation == "gamma_shape":
                g["shape"] = [256, 1]
            elif mutation == "gamma_dtype":
                g["dtype"] = "f32"
            elif mutation == "gamma_bound":
                g["abs_bound"] = float("inf")
            elif mutation == "producer":
                n["inputs"] = []
            elif mutation == "axis":
                n["dataflow"]["axis"] = "x"
            else:
                n["dataflow"]["unproven"] = "true"
            with self.subTest(mutation=mutation), self.assertRaises(Error):
                mean_boundary(n, x, g, derived_input_bound=34.5625)
