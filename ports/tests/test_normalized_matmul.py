"""Composition rejects incompatible ownership and preserves staged numerical meaning."""

import copy, sys, unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "toolchain"))
from frontend import parse, Error
from ir import verify
from mesh_normalized_matmul import plan, reference
from normalized_matmul_debug import inspect


class ResidentNormalization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = parse(ROOT / "projects/waferllm/normalized_matmul_64x128_8x8/hls.cpp")
        cls.m = verify(cls.raw, 6, 1)

    def test_incompatible_subgraph_rejected(self):
        for mutation in [
            lambda m: m["nodes"][4].update(
                inputs=[m["nodes"][0]["id"], m["nodes"][3]["id"]]
            ),
            lambda m: m["nodes"][4]["dataflow"].update(cols=4),
            lambda m: m["nodes"][4]["dataflow"].update(initial_align="bidirectional"),
            lambda m: m["nodes"][3].update(shape=[128, 64]),
            lambda m: m["nodes"][3].update(host="x"),
            lambda m: m["nodes"][3].update(dtype="f32"),
        ]:
            v = copy.deepcopy(self.raw)
            mutation(v)
            with self.assertRaises(Error):
                verify(v, 6, 1)

    def test_packing_and_memory_limits(self):
        m = copy.deepcopy(self.raw)
        for i in (0, 2, 4):
            m["nodes"][i]["shape"] = [24, 24]
        m["nodes"][1]["shape"] = [1, 24]
        m["nodes"][3]["shape"] = [24, 24]
        with self.assertRaises(Error):
            verify(m, 6, 1)
        m = copy.deepcopy(self.raw)
        for i in (0, 2, 4):
            m["nodes"][i]["shape"] = [256, 512]
        m["nodes"][1]["shape"] = [1, 512]
        m["nodes"][3]["shape"] = [512, 512]
        with self.assertRaises(Error):
            verify(m, 6, 1)

    def test_counter_storage_selects_admissible_large_shape(self):
        m = copy.deepcopy(self.raw)
        for i in (0, 2, 4):
            m["nodes"][i]["shape"] = [128, 512]
        m["nodes"][1]["shape"] = [1, 512]
        m["nodes"][3]["shape"] = [512, 512]
        with self.assertRaises(Error):
            verify(m, 6, 1)
        m["instrumentation"] = "counters"
        s = plan(verify(m, 6, 1))
        self.assertLess(sum(s["memory_per_pe"].values()), 49152)
        self.assertEqual(s["memory_per_pe"]["observation_half_arrays"], 4)

    def test_identity_projection_and_prefix(self):
        s = plan(self.m)
        x = np.tile(np.array([-0.25, 0.5, 0.25, -0.5]), (64, 32))
        w = np.ones((1, 128))
        normalized, history, actual = reference(s, x, w, np.eye(128))
        np.testing.assert_array_equal(normalized, actual)
        for y in range(8):
            for col in range(8):
                np.testing.assert_array_equal(
                    history[y, col, -1],
                    actual[y * 8 : y * 8 + 8, col * 16 : col * 16 + 16],
                )
        self.assertEqual(s["resources"]["input_queues"], [3, 4, 5, 6, 7])
        self.assertIn("synchronous RMS completion", s["buffers"]["scratch"])

    def test_renamed_user_symbols(self):
        m = copy.deepcopy(self.raw)
        mapping = {n["id"]: "renamed_" + str(i) for i, n in enumerate(m["nodes"])}
        for n in m["nodes"]:
            n["id"] = mapping[n["id"]]
            n["inputs"] = [mapping[v] for v in n["inputs"]]
            if "host" in n:
                n["host"] = "new_" + n["host"]
        self.assertEqual(plan(verify(m, 6, 1)), plan(self.m))

    def test_independent_input_declaration_order(self):
        m = copy.deepcopy(self.raw)
        m["nodes"] = [
            m["nodes"][3],
            m["nodes"][1],
            m["nodes"][0],
            m["nodes"][2],
            m["nodes"][4],
            m["nodes"][5],
        ]
        self.assertEqual(verify(m, 6, 1), self.m)
        m["nodes"][0]["inputs"] = [m["nodes"][2]["id"]]
        with self.assertRaises(Error):
            verify(m, 6, 1)


class ResidentDebug(unittest.TestCase):
    def test_bounds_and_unobserved_counter(self):
        m = verify(
            parse(ROOT / "projects/waferllm/normalized_matmul_64x128_8x8/hls.cpp"), 6, 1
        )
        s = plan(m)
        self.assertEqual(len(inspect(s, None, "p3_2", 0, 8)["k_blocks_consumed"]), 8)
        self.assertEqual(inspect(s, None, "p3_2", 0, 0)["operation"], "rmsnorm")
        for node, epoch, step in [("p8_0", 0, 0), ("p0_0", 6, 0), ("p0_0", 0, 9)]:
            with self.assertRaises(Error):
                inspect(s, None, node, epoch, step)
        s["instrumentation"] = "counters"
        self.assertFalse(inspect(s, None, "p0_0", 0, 0)["observed"])
        with self.assertRaises(Error):
            verify(m, 6, 2)
