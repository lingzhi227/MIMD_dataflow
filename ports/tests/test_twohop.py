"""Compare generic cycle planning with pinned host permutation and typed bounds."""

import copy, importlib.util, sys, unittest, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "toolchain"))
from frontend import parse
from ir import verify
from mesh_twohop import plan, pack, block_index


class TwoHop(unittest.TestCase):
    def test_original_host_permutation_and_cycle(self):
        spec = importlib.util.spec_from_file_location(
            "wafer_host", ROOT / "references/waferllm-host/MeshGEMM/host_common.py"
        )
        hc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hc)
        for p in (4, 8):
            state = np.random.get_state()
            np.random.seed(181)
            try:
                a, b, offset = hc.make_inputs(p, 64, 64, 64)
            finally:
                np.random.set_state(state)
            ap, bp = pack(a, b, p)
            t = 64 // p
            np.testing.assert_array_equal(ap.ravel(), hc.tile_X(a, p, t, t))
            np.testing.assert_array_equal(bp.ravel(), hc.tile_W(offset, p, t, t))
            ind = np.zeros((p, p), dtype=int)
            ind[0] = np.arange(p)
            for y in range(1, p):
                for x in range(p):
                    if y == 1:
                        ind[y, x] = hc.assignId(ind[0, x], p)[1]
                    elif (y - 1) % 2 == 0:
                        ind[y, x] = hc.assignId(ind[y - 2, x], p)[1]
                    else:
                        ind[y, x] = hc.assignId(ind[y - 2, x], p)[0]
            xblocks = ind.copy()
            wblocks = ind.copy()
            recv = [hc.assignId(i, p)[1] for i in range(p)]
            for step in range(p):
                for y in range(p):
                    for x in range(p):
                        self.assertEqual(xblocks[y, x], wblocks[y, x])
                        self.assertEqual(xblocks[y, x], block_index(p, y, x, step))
                xblocks = xblocks[:, recv]
                wblocks = wblocks[recv, :]

    def test_debug_half_bits_and_warm_round(self):
        import json
        from twohop_debug import inspect

        m = verify(
            parse(ROOT / "projects/waferllm/mesh_twohop_64_4x4_f16/hls.cpp"), 6, 2
        )
        s = plan(m)
        self.assertFalse(inspect(s, None, "p3_2", 5, 3)["observed"])
        for node, epoch, step in [("p4_0", 0, 0), ("p0_0", 6, 0), ("p0_0", 0, 4)]:
            with self.assertRaises(ValueError):
                inspect(s, None, node, epoch, step)
        shape = lambda v: [[v for x in range(4)] for y in range(4)]
        d = dict(
            history_bits=shape([[1, 32768, 15360] for i in range(4)]),
            witness_bits=shape([[15360, 0, 0, 15360] for i in range(4)]),
            timing=shape([[65535, 0, 0, 9, 1, 0] for i in range(4)]),
            progress=shape([4, 4, 4, 4, 2, 3, 6]),
            queue=shape([60, 60]),
        )
        view = inspect(s, dict(diagnostics=[d] * 6), "p3_2", 5, 3)
        self.assertEqual(view["prefix_values"], [2**-24, -0.0, 1.0])
        self.assertEqual(view["compute_cycles"], 10)
        self.assertEqual(view["progress"]["warm_entries"], 6)

    def test_counter_mode_preserves_arithmetic_and_drops_history(self):
        from mesh_twohop import generate

        m = verify(
            parse(ROOT / "projects/waferllm/mesh_twohop_64_4x4_f16/hls.cpp"), 6, 2
        )
        full = plan(m)
        m["instrumentation"] = "counters"
        light = plan(m)
        self.assertEqual(full["resources"], light["resources"])
        self.assertLess(
            sum(light["memory_per_pe"].values()), sum(full["memory_per_pe"].values())
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            generate(full, root)
            original = (root / "pe.csl").read_text()
            generate(light, root)
            counters = (root / "pe.csl").read_text()
            self.assertNotIn("@fmovh(target,res_dsd)", counters)
            self.assertIn("@fmach", counters)
            # Restrict changes to history instrumentation: communication and
            # DSR compute instructions must survive unchanged.
            for token in [
                "comm_mod.two_hop_comm(ptr_X_send, ptr_W_send, ptr_X_recv, ptr_W_recv);",
                "@fmach",
                "@load_to_dsr",
                "@activate(next_step_id)",
            ]:
                self.assertEqual(original.count(token), counters.count(token))
        m["instrumentation"] = "unknown"
        with self.assertRaises(ValueError):
            plan(m)

    def test_typed_resource_and_layout_constraints(self):
        raw = parse(ROOT / "projects/waferllm/mesh_twohop_64_4x4_f16/hls.cpp")
        m = verify(raw, 6, 2)
        s = plan(m)
        self.assertEqual(m["nodes"][-1]["dtype"], "f16")
        self.assertEqual(s["resources"]["microthreads"], [1, 2, 3, 4])
        self.assertEqual(s["resources"]["compute_dsr"], 1)
        for field, value in [("cols", 8), ("overlap", "none")]:
            bad = copy.deepcopy(raw)
            bad["nodes"][2]["dataflow"][field] = value
            with self.assertRaises(ValueError):
                verify(bad, 6, 2)
        bad = copy.deepcopy(raw)
        for node in bad["nodes"][:3]:
            node["shape"] = [4, 4]
        with self.assertRaisesRegex(ValueError, "even tile"):
            verify(bad, 6, 2)
        bad = copy.deepcopy(raw)
        bad["nodes"][0]["dtype"] = "f32"
        with self.assertRaises(ValueError):
            verify(bad, 6, 2)


if __name__ == "__main__":
    unittest.main()
