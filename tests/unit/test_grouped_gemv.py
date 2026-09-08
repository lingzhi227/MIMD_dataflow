"""Pinned host packing and independently enumerated grouped reduction contracts."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import copy, importlib.util, sys, unittest, tempfile
from pathlib import Path
import numpy as np

ROOT = repository_root(__file__)
sys.path.insert(0, str(ROOT / "lib"))
from frontend import parse
from ir import verify
from mesh_grouped_gemv import plan, pack, reference, reduce_group


class GroupedGemv(unittest.TestCase):
    def test_host_replication_and_matrix_order(self):
        spec = importlib.util.spec_from_file_location(
            "grouped_host", ROOT / "third_party/references/waferllm-host/MeshGEMV/host_common.py"
        )
        h = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(h)
        state = np.random.get_state()
        np.random.seed(2718)
        try:
            for p in (4, 8):
                x, xrep, w = h.make_inputs(p, 128, 128)
                actual_x, actual_w = pack(x, w, p)
                np.testing.assert_array_equal(actual_x.ravel(), xrep.ravel())
                np.testing.assert_array_equal(
                    actual_w.ravel(), h.tile_W(w, p, 128 // p, 128 // p)
                )
        finally:
            np.random.set_state(state)

    def test_roots_and_half_tree(self):
        # Root at1 first consumes the complete lower chain, then upper0.
        v = np.asarray([1, 2048, -2048, 1], dtype=float)
        h = lambda a, b: float(np.float16(a + b))
        expected = h(v[0], h(h(v[3], v[2]), v[1]))
        self.assertEqual(float(reduce_group(v, 1)), expected)
        for p, g in [(4, 2), (8, 2), (8, 4)]:
            file = (
                ROOT
                / f"benchmarks/linear_algebra/waferllm/mesh_grouped_gemv_{128 if p==4 else 512}_{p}x{p}_g{g}_f16/hls.cpp"
            )
            m = verify(parse(file), 6, 2)
            s = plan(m)
            self.assertEqual(s["root_within_group"], (p // g) // 2)
            self.assertEqual(s["global_root"], (g // 2) * (p // g) + (p // g) // 2)
            a = np.ones((1, s["M"]))
            b = np.ones((s["M"], s["N"]))
            local, groups, total = reference(s, a, b)
            np.testing.assert_array_equal(local, s["Mt"])
            np.testing.assert_array_equal(groups, s["Mt"] * s["group_size"])
            np.testing.assert_array_equal(total, s["M"])
            self.assertEqual(s["resources"]["input_queues"], list(range(2, 8)))

    def test_debug_inactive_phase_is_not_claimed(self):
        from grouped_gemv_debug import inspect

        m = verify(
            parse(ROOT / "benchmarks/linear_algebra/waferllm/mesh_grouped_gemv_128_4x4_g2_f16/hls.cpp"),
            8,
            2,
        )
        s = plan(m)
        self.assertFalse(inspect(s, None, "p0_0", 7, 1)["phase_record_active"])
        self.assertTrue(inspect(s, None, "p0_1", 7, 1)["phase_record_active"])
        self.assertFalse(inspect(s, None, "p0_1", 7, 2)["phase_record_active"])
        self.assertTrue(inspect(s, None, "p0_3", 7, 2)["phase_record_active"])
        for node, epoch, phase in [("p4_0", 0, 0), ("p0_0", 8, 0), ("p0_0", 0, 3)]:
            with self.assertRaises(ValueError):
                inspect(s, None, node, epoch, phase)

    def test_counters_preserve_compute_and_omit_sample_records(self):
        import tempfile
        from mesh_grouped_gemv import generate

        m = verify(
            parse(ROOT / "benchmarks/linear_algebra/waferllm/mesh_grouped_gemv_128_4x4_g2_f16/hls.cpp"),
            8,
            2,
        )
        original = plan(m)
        m["instrumentation"] = "counters"
        lean = plan(m)
        self.assertEqual(original["resources"], lean["resources"])
        self.assertLess(
            lean["memory_per_pe"]["result_and_observations"],
            original["memory_per_pe"]["result_and_observations"],
        )
        with tempfile.TemporaryDirectory() as td:
            generate(lean, td)
            text = (Path(td) / "pe.csl").read_text()
            self.assertIn("@map(gemv_static_step, X_dsd);", text)
            self.assertIn("comm_mod.two_tree_allreduce_y(ptr_res);", text)
            self.assertNotIn("@fmovh(d,res_dsd);", text)
            self.assertNotIn("timestamp.get_timestamp(&hls_compute_start)", text)
        m["instrumentation"] = "invalid"
        with self.assertRaises(ValueError):
            plan(m)

    def test_fail_closed_groups_and_dtype(self):
        raw = parse(ROOT / "benchmarks/linear_algebra/waferllm/mesh_grouped_gemv_128_4x4_g2_f16/hls.cpp")
        for key, value in [
            ("groups", 1),
            ("groups", 4),
            ("groups", 3),
            ("cols", 8),
            ("rows", 4.0),
            ("result", "root_only"),
        ]:
            bad = copy.deepcopy(raw)
            bad["nodes"][2]["dataflow"][key] = value
            with self.assertRaises(ValueError):
                verify(bad, 6, 2)
        bad = copy.deepcopy(raw)
        bad["nodes"][1]["dtype"] = "f32"
        with self.assertRaises(ValueError):
            verify(bad, 6, 2)


if __name__ == "__main__":
    unittest.main()
