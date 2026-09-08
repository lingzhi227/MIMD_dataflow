import copy, sys, tempfile, unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "toolchain")]
from frontend import parse, Error
from ir import verify
from mesh_cache_attention import plan, generate
from mesh_cache_attention_sdk import packed, unshard
from cache_attention_fixtures import batches


class CacheAttention(unittest.TestCase):
    def test_direct_half_underflow_sign_matches_local_fast_path(self):
        from cache_attention_reference import matmul

        a = np.asarray([[0, -(2**-24)], [0, 2**-24]])
        w = np.asarray([[0, 0], [0.5, 0.5]])
        direct = matmul(a, w, 2).astype(np.float16).view(np.uint16)
        blocked = matmul(a, w, 1).astype(np.float16).view(np.uint16)
        np.testing.assert_array_equal(direct, [[32768, 32768], [0, 0]])
        np.testing.assert_array_equal(blocked, np.zeros((2, 2), np.uint16))

    def raw(self):
        return parse(ROOT / "projects/waferllm/cache_attention_5x256x512_8x8/hls.cpp")

    def test_structural_layout_and_resource_contract(self):
        m = verify(self.raw(), 8, 2)
        s = plan(m)
        self.assertEqual((s["B"], s["N"], s["S"], s["P"]), (5, 256, 512, 8))
        self.assertEqual(sum(s["memory_per_pe"].values()), 40212)
        self.assertEqual(s["numeric_allocations"]["max_gathered"], 192)
        self.assertEqual(s["resources"]["local_tasks"], [10, 11, 14, 15, 16, 17])
        self.assertTrue(s["storage_lifetimes"]["validation"]["checked"])
        r = self.raw()
        ids = {n["id"]: f"node{i}" for i, n in enumerate(r["nodes"])}
        for n in r["nodes"]:
            n["id"] = ids[n["id"]]
            n["inputs"] = [ids[i] for i in n["inputs"]]
            if "host" in n:
                n["host"] = "port_" + n["host"]
        r["nodes"].reverse()
        ss = plan(verify(r, 8, 2))
        with tempfile.TemporaryDirectory() as td:
            a, b = Path(td) / "a", Path(td) / "b"
            a.mkdir()
            b.mkdir()
            generate(s, a)
            generate(ss, b)
            self.assertEqual(
                {p.name: p.read_bytes() for p in a.iterdir()},
                {p.name: p.read_bytes() for p in b.iterdir()},
            )

    def test_every_cache_word_uses_device_axes(self):
        m = verify(self.raw(), 8, 2)
        s = plan(m)
        batch = batches(5, 256, 512)[4]
        raw = packed(s, m, batch)
        key = np.asarray(batch["key"]).reshape(512, 256)
        value = np.asarray(batch["value"]).reshape(512, 256)
        for y in range(8):
            for x in range(8):
                # Independent scalar indexing, including local flattening.
                for i in range(32):
                    for j in range(64):
                        self.assertEqual(
                            raw["K"][y, x, i * 64 + j], key[y * 64 + j, x * 32 + i]
                        )
                        self.assertEqual(
                            raw["V"][y, x, j * 32 + i], value[y * 64 + j, x * 32 + i]
                        )
        np.testing.assert_array_equal(
            unshard(s, raw["Q"], "x", 256), np.asarray(batch["query"]).reshape(5, 256)
        )
        np.testing.assert_array_equal(
            unshard(s, raw["X"], "y", 256), np.asarray(batch["x"]).reshape(5, 256)
        )
        # Recovered upstream feature-Y / sequence-X packing must be distinguishable.
        wrong = (
            np.asarray(batch["key"])
            .reshape(512, 256)
            .T.reshape(8, 32, 8, 64)
            .transpose(0, 2, 1, 3)
            .reshape(8, 8, -1)
        )
        self.assertGreater(np.count_nonzero(wrong != raw["K"]), 10000)

    def test_invalid_edges_precision_domain_and_capacity_rejected(self):
        for change in [
            "axis",
            "precision",
            "edge",
            "bound",
            "scale",
            "extent",
            "block",
        ]:
            r = self.raw()
            by = {n["id"]: n for n in r["nodes"]}
            if change == "axis":
                by["score"]["dataflow"]["axis"] = "y"
            if change == "precision":
                by["probability"]["dataflow"]["collective"] = "f16"
            if change == "edge":
                by["delta"]["inputs"][0] = "probability"
            if change == "bound":
                by["query"]["abs_bound"] = 2
            if change == "scale":
                by["probability"]["scale"] = 0.1
            if change == "block":
                by["context"]["block_size"] = 17
            if change == "extent":
                by["key"]["shape"] = [1024, 256]
            with self.assertRaises(Error, msg=change):
                verify(r, 8, 2)

    def test_blocked_range_does_not_use_long_half_recurrence(self):
        from blocked_matmul import finite_bound
        from input_contracts import half_dot_bound
        from binary16 import quantize

        a = quantize(0.1)
        self.assertEqual(finite_bound(a, a, 64, 32), 0.64013671875)
        self.assertEqual(half_dot_bound(a, a, 64), 0.63671875)
        self.assertGreater(finite_bound(a, a, 64, 32), half_dot_bound(a, a, 64))

    def test_max_and_sum_workspace_lease_reuse_is_serial(self):
        s = plan(verify(self.raw(), 8, 2))
        life = s["storage_lifetimes"]
        self.assertEqual(
            [v["first"] for v in life["values"] if v["storage"] == "collective_send"],
            [1, 5, 7, 9],
        )
        self.assertEqual(
            [v["first"] for v in life["values"] if v["storage"] == "max_gathered"], [3]
        )
        self.assertEqual(
            {v["name"] for v in life["values"] if v.get("immutable")},
            {"X", "Q", "K", "V", "W"},
        )


if __name__ == "__main__":
    unittest.main()
