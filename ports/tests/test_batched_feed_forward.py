import copy, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "toolchain"))
from frontend import parse, Error
from ir import verify
from mesh_batched_feed_forward import plan, generate
from region_lifetimes import verify as lifetimes


class BatchedFFN(unittest.TestCase):
    def raw(self):
        return parse(
            ROOT / "projects/waferllm/batched_feed_forward_5x256x512_8x8/hls.cpp"
        )

    def test_plan_resources_and_structural_identity(self):
        m = verify(self.raw(), 8, 2)
        s = plan(m)
        self.assertEqual(s["memory_per_pe"]["weights"], 12288)
        self.assertEqual(sum(s["memory_per_pe"].values()), 40676)
        self.assertTrue(s["storage_lifetimes"]["validation"]["checked"])
        r = self.raw()
        ids = {n["id"]: f"v{i}" for i, n in enumerate(r["nodes"])}
        for n in r["nodes"]:
            n["id"] = ids[n["id"]]
            n["inputs"] = [ids[i] for i in n["inputs"]]
            if "host" in n:
                n["host"] = "port_" + n["host"]
        r["nodes"].reverse()
        ss = plan(verify(r, 8, 2))
        with tempfile.TemporaryDirectory() as td:
            a = Path(td) / "a"
            b = Path(td) / "b"
            a.mkdir()
            b.mkdir()
            generate(s, a)
            generate(ss, b)
            self.assertEqual(
                {p.name: p.read_bytes() for p in a.iterdir()},
                {p.name: p.read_bytes() for p in b.iterdir()},
            )

    def test_precision_shape_and_range_rejections(self):
        for op, key, value in [
            ("rmsnorm", "collective", "f16"),
            ("silu", "math", "sdk_half"),
            ("add", "axis", "x"),
        ]:
            r = self.raw()
            next(n for n in r["nodes"] if n["op"] == op)["dataflow"][key] = value
            with self.assertRaises(Error):
                verify(r, 8, 2)
        r = self.raw()
        next(n for n in r["nodes"] if n.get("host") == "wd")["abs_bound"] = 2
        with self.assertRaises(Error):
            verify(r, 8, 2)
        r = self.raw()
        next(n for n in r["nodes"] if n.get("host") == "wd")["shape"] = [256, 512]
        with self.assertRaises(Error):
            verify(r, 8, 2)

    def test_callback_lifetimes_and_immutable_input(self):
        s = plan(verify(self.raw(), 8, 2))["storage_lifetimes"]
        bad = copy.deepcopy(s)
        next(v for v in bad["values"] if v["name"] == "collective_send_1")["last"] = 3
        with self.assertRaises(Error):
            lifetimes(bad["storage"], bad["values"], bad["phases"])
        bad = copy.deepcopy(s)
        bad["phases"][1]["release"] = []
        with self.assertRaises(Error):
            lifetimes(bad["storage"], bad["values"], bad["phases"])
        bad = copy.deepcopy(s)
        next(v for v in bad["values"] if v["name"] == "X")["last"] = 6
        with self.assertRaises(Error):
            lifetimes(bad["storage"], bad["values"], bad["phases"])


class BatchedAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        from batched_ffn_reference import reference
        from mesh_batched_feed_forward_sdk import packed, values, decode, extents

        text = (
            (ROOT / "projects/waferllm/batched_feed_forward_5x256x512_8x8/hls.cpp")
            .read_text()
            .replace("5,256", "1,32")
            .replace("1,256", "1,32")
            .replace("256,512", "32,32")
            .replace("512,256", "32,32")
            .replace("rows=8 cols=8", "rows=4 cols=4")
        )
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "source.cpp"
            p.write_text(text)
            cls.m = verify(parse(p), 2, 2)
        cls.s = plan(cls.m)
        rng = np.random.default_rng(817)
        b = {
            n["host"]: np.asarray(
                rng.uniform(-n["abs_bound"], n["abs_bound"], n["shape"]), np.float16
            )
            .astype(float)
            .ravel()
            .tolist()
            for n in cls.m["nodes"][:5]
        }
        cls.batches = [b, b]
        raw, _ = reference(cls.s, values(cls.m, b))
        raw.update(packed(cls.s, cls.m, b))
        d = {
            k: np.asarray(v, np.float16).view(np.uint16).tolist()
            for k, v in raw.items()
        }
        d.update(
            progress=np.tile([1] * 8, (4, 4, 1)).tolist(),
            queues=np.full((4, 4, 2), 60).tolist(),
            timing=np.tile([100, 0, 0, 200, 0, 0], (4, 4, 1)).tolist(),
        )
        assert set(d) == set(extents(cls.s))
        cls.r = dict(
            success=False,
            runtime_instances=1,
            cases=[decode(cls.s, cls.m, d)],
            diagnostics=[d],
            launches=["hls_main"],
        )

    def test_partial_is_not_full(self):
        from batched_ffn_reference import audit_cases

        report = audit_cases(
            self.s, self.m, self.batches, self.r, require_complete=False
        )
        self.assertFalse(report["full_run_passed"])
        with self.assertRaises(Error):
            audit_cases(self.s, self.m, self.batches, self.r)
        r = copy.deepcopy(self.r)
        r["success"] = True
        with self.assertRaises(Error):
            audit_cases(self.s, self.m, self.batches, r, require_complete=False)

    def test_saved_state_and_timer_damage_rejected(self):
        from batched_ffn_reference import audit_cases

        for port in (
            "X",
            "gamma",
            "weights",
            "normalized",
            "scratch",
            "sums",
            "history",
            "partial",
            "projections",
            "activation",
            "hidden",
            "down_partial",
            "delta",
            "result",
            "progress",
            "queues",
        ):
            r = copy.deepcopy(self.r)
            r["diagnostics"][0][port][3][3][0] ^= 4
            with self.assertRaises((Error, AssertionError), msg=port):
                audit_cases(self.s, self.m, self.batches, r, require_complete=False)
        r = copy.deepcopy(self.r)
        r["diagnostics"][0]["timing"][3][3][5] = 65535
        with self.assertRaises(Error):
            audit_cases(self.s, self.m, self.batches, r, require_complete=False)

    def test_debugger_reads_actual_words_and_reports_missing(self):
        from batched_ffn_debug import inspect

        d = inspect(self.s, self.r, "p3_3", 0, 8)
        self.assertEqual(d["raw_words"], self.r["diagnostics"][0]["delta"][3][3])
        self.assertFalse(inspect(self.s, self.r, "p3_3", 1, 8)["available"])
        s = dict(self.s, instrumentation="counters")
        self.assertFalse(inspect(s, self.r, "p3_3", 0, 7)["observed"])


if __name__ == "__main__":
    unittest.main()
