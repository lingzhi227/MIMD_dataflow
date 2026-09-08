
from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)

import copy, json, sys, tempfile, unittest
from pathlib import Path
import numpy as np

ROOT = repository_root(__file__)
sys.path[:0] = [str(ROOT), str(ROOT / "lib")]
from frontend import parse, Error
from ir import verify
from mesh_projected_cache import plan, generate
from mesh_projected_cache_sdk import packed, extents
from projected_cache_fixtures import batches
from blocked_projection_bounds import bound


class ProjectedCache(unittest.TestCase):
    def raw(self):
        return parse(
            ROOT / "benchmarks/inference/waferllm/projected_cache_attention_3x256x512_8x8/hls.cpp"
        )

    def test_whole_graph_plan_and_renaming(self):
        m = verify(self.raw(), 8, 2)
        s = plan(m)
        self.assertEqual(
            (len(m["nodes"]), s["projection_blocks"], len(s["stages"])),
            (25, [4, 32, 32], 16),
        )
        self.assertEqual(sum(s["memory_per_pe"].values()), 46816)
        self.assertTrue(s["storage_lifetimes"]["validation"]["checked"])
        self.assertEqual(
            s["resources"]["extent_transitions"], [4, 288, 192, 4, 4, 96, 96]
        )
        self.assertGreater(s["numerical_bounds"]["derived_query"], 1)
        self.assertLess(s["numerical_bounds"]["context"], 1.1)
        r = self.raw()
        ids = {n["id"]: f"renamed{i}" for i, n in enumerate(r["nodes"])}
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
                {f.name: f.read_bytes() for f in a.iterdir()},
                {f.name: f.read_bytes() for f in b.iterdir()},
            )

    def test_original_and_auxiliary_connectivity_rejected(self):
        for change in (
            lambda r: r["nodes"][23].update(host=r["nodes"][22]["host"]),
            lambda r: r["nodes"][24].update(host=""),
            lambda r: r["nodes"][13].update(
                inputs=[r["nodes"][11]["id"], r["nodes"][4]["id"]]
            ),
            lambda r: r["nodes"][15].update(
                inputs=[r["nodes"][12]["id"], r["nodes"][6]["id"], r["nodes"][5]["id"]]
            ),
            lambda r: r["nodes"][17].update(
                inputs=[r["nodes"][15]["id"], r["nodes"][16]["id"]]
            ),
            lambda r: r["nodes"][11].update(block_size=3),
            lambda r: r["nodes"][14]["dataflow"].update(axis="y"),
        ):
            r = self.raw()
            change(r)
            with self.assertRaises(Error):
                verify(r, 8, 2)

    def test_packing_matches_each_role_scalar_indices(self):
        m = verify(self.raw(), 8, 2)
        s = plan(m)
        batch = batches(3, 256, 512)[4]
        raw = packed(s, m, batch)
        self.assertEqual(
            set(raw), {"X", "gamma", "qkv_weights", "cosine", "sine", "K", "V", "W"}
        )
        for y, x in ((0, 0), (0, 7), (7, 0), (3, 5)):
            for row in range(3):
                for j in range(32):
                    self.assertEqual(
                        raw["X"][y, x, row * 32 + j], batch["x"][row * 256 + y * 32 + j]
                    )
            for t, name in enumerate(("wq", "wk", "wv")):
                for i in range(32):
                    for j in range(32):
                        self.assertEqual(
                            raw["qkv_weights"][y, x, t * 1024 + i * 32 + j],
                            batch[name][(y * 32 + i) * 256 + x * 32 + j],
                        )
            for j in range(64):
                for i in range(32):
                    self.assertEqual(
                        raw["K"][y, x, i * 64 + j],
                        batch["key"][(y * 64 + j) * 256 + x * 32 + i],
                    )
        self.assertEqual(extents(s)["progress"], 11)

    def test_blocked_l1_certificate_guard_and_tightening(self):
        plain = bound(258.629, 0.03125, 32, 8, 32)
        short = bound(258.629, 0.03125, 32, 8, 4)
        self.assertLess(short["reduced"], plain["reduced"])
        self.assertGreaterEqual(short["reduced"], short["native"])
        for block in (0, 3, 33):
            with self.assertRaises(Error):
                bound(258.629, 0.03125, 32, 8, block)

    def test_projection_certificate_rejects_inexact_geometry(self):
        for l1, local, participants in (
            (258.629, 32.0, 8),
            (258.629, 32, 8.0),
            (258.629, True, 8),
            (float("nan"), 32, 8),
            (float("inf"), 32, 8),
            (-1, 32, 8),
        ):
            with self.assertRaises(Error):
                bound(l1, 0.03125, local, participants, 4)


if __name__ == "__main__":
    unittest.main()
