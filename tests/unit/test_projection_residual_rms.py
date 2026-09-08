"""Development composition contracts; no SDK qualification is implied by these tests."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import copy, sys, tempfile, unittest
from pathlib import Path
import numpy as np

ROOT = repository_root(__file__)
sys.path[:0] = [str(ROOT / "lib"), str(ROOT / "experiments")]
from build_projection_residual_rms import source
from frontend import parse, Error
from mesh_projection_residual_rms import verify, plan, evaluate
from rms_bounds import row_norm_bound, inverse_bound
from mesh_rms import reference as rms_reference


class ProjectionResidualRMS(unittest.TestCase):
    def module(self, text=None):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "hls.cpp"
            p.write_text(text or source())
            return parse(p)

    def test_roles_shapes_resources_and_range(self):
        m = verify(self.module(), 8, 2)
        s = plan(m)
        self.assertFalse(s["ownership"]["intermediate_host_transfer"])
        self.assertEqual(
            [v["operation"] for v in s["stages"]], ["matmul", "add", "rmsnorm"]
        )
        self.assertLess(sum(s["memory_per_pe"].values()), 49152)
        self.assertEqual(s["numerical_bounds"]["residual_sum"], 1.5)
        self.assertLess(s["numerical_bounds"]["normalization"]["output"], 65504)
        self.assertEqual(s["resources"]["colors"], list(range(1, 12)))

    def test_commuted_add_and_source_ids_do_not_dispatch(self):
        m = self.module()
        names = {n["id"]: "value_" + n["id"] for n in m["nodes"]}
        for n in m["nodes"]:
            n["id"] = names[n["id"]]
            n["inputs"] = [names[i] for i in n["inputs"]]
        next(n for n in m["nodes"] if n["op"] == "add")["inputs"].reverse()
        self.assertEqual(verify(m, 8, 2)["profile"], "mesh_projection_residual_rms.v1")

    def test_bad_graph_policy_and_range_are_rejected(self):
        for mutate in [
            lambda m: next(n for n in m["nodes"] if n["op"] == "add")[
                "dataflow"
            ].update(rows=4),
            lambda m: (
                m["nodes"][0].update(abs_bound=2),
                m["nodes"][1].update(abs_bound=2),
            ),
            lambda m: next(n for n in m["nodes"] if n["op"] == "matmul")[
                "dataflow"
            ].update(accumulation="block_f32"),
            lambda m: next(n for n in m["nodes"] if n["op"] == "add")[
                "inputs"
            ].__setitem__(1, m["nodes"][0]["id"]),
        ]:
            with self.assertRaises(Error):
                m = self.module()
                mutate(m)
                verify(m, 8, 2)

    def test_debugger_distinguishes_unobserved_and_unavailable(self):
        from projection_residual_rms_debug import inspect

        s = plan(verify(self.module(), 8, 2))
        s["instrumentation"] = "counters"
        p, mt, l = s["P"], s["Mt"], s["length"]
        tile = lambda values: [[list(values) for _ in range(p)] for _ in range(p)]
        d = dict(
            progress=tile([1, p, 1, 1]),
            timing=tile([1, 0, 0, 2, 0, 0]),
            queues=tile([248, 248]),
            result=tile(range(l)),
            inverse=tile(range(mt)),
        )
        r = dict(diagnostics=[d])
        prefix = inspect(s, r, "p2_3", 0, 0)
        self.assertTrue(prefix["available"])
        self.assertFalse(prefix["observed"])
        self.assertIsNone(prefix["half_bits"])
        final = inspect(s, r, "p2_3", 0, p + 1)
        self.assertEqual(final["half_bits"], list(range(l)))
        self.assertEqual(final["row_inverse_half_bits"], list(range(mt)))
        self.assertFalse(inspect(s, r, "p2_3", 1, p + 1)["available"])
        with self.assertRaises(Error):
            inspect(s, r, "p8_0", 0, 0)

    def test_correlated_range_covers_underflow_and_concentrated_rows(self):
        bound = row_norm_bound(1.5, 1.5, 8, 8, 1e-6)
        # Independent real RMS scale is sqrt(64)*1.5=12; source half rounding
        # and approximate math require a little slack, not inverse(0)*max(x).
        self.assertLess(bound["output"], 12.1)
        self.assertGreaterEqual(bound["output"], 12.0)
        s = dict(rows=4, cols=8, M=8, N=64, Mt=2, Nt=8, epsilon=1e-6)
        x = np.zeros((8, 64))
        x[:, 0] = [0, 2**-24, 2**-14, 2**-13, 2**-12, 0.125, -1.5, 1.5]
        for gamma in (1.5, -1.5):
            *_, y = rms_reference(s, x, np.full((1, 64), gamma))
            self.assertLessEqual(float(np.max(np.abs(y))), bound["output"])
        with self.assertRaises(Error):
            from rms_bounds import correlated_output_bound

            correlated_output_bound(1.5, 1.5, 1.0, 64, 1e-6)

    def test_range_tree_matches_constant_positive_target(self):
        for cols in (4, 8):
            bound = row_norm_bound(1.5, 1.5, 8, cols, 1e-6)
            s = dict(rows=4, cols=cols, M=16, N=8 * cols, Mt=4, Nt=8, epsilon=1e-6)
            local, total, inv, y = rms_reference(
                s, np.full((16, 8 * cols), 1.5), np.full((1, 8 * cols), 1.5)
            )
            self.assertEqual(float(local.max()), bound["local_square_sum"])
            self.assertEqual(float(total.max()), bound["reduced_square_sum"])
            self.assertLessEqual(float(inv.max()), bound["inverse"])
            self.assertLessEqual(float(y.max()), bound["output"])


if __name__ == "__main__":
    unittest.main()
