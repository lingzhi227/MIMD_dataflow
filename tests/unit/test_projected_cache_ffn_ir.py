
from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)

import copy, sys, unittest
from pathlib import Path

ROOT = repository_root(__file__)
sys.path.insert(0, str(ROOT / "lib"))
from frontend import parse, Error
from projected_cache_ffn_ir import canonical


class ComposedAttentionFFN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = parse(
            ROOT / "benchmarks/inference/waferllm/projected_cache_ffn_3x256x512x512_16x16/hls.cpp"
        )

    def test_shared_contracts(self):
        c = canonical(self.m)
        self.assertEqual(len(c["attention"]["nodes"]), 25)
        self.assertEqual(len(c["ffn"]["nodes"]), 13)
        self.assertEqual(c["boundary"]["gamma"], "gamma")
        self.assertTrue(c["boundary"]["virtual_input_requires_parent_range"])

    def test_order_and_names_do_not_dispatch(self):
        m = copy.deepcopy(self.m)
        names = {n["id"]: "v" + str(i) for i, n in enumerate(m["nodes"])}
        for n in m["nodes"]:
            n["id"] = names[n["id"]]
            n["inputs"] = [names[i] for i in n["inputs"]]
        m["nodes"].reverse()
        self.assertEqual(canonical(m)["boundary"]["producer"], names["result"])

    def test_semantic_misconnections(self):
        for label in ("residual", "gamma", "norm", "gate", "output", "extra"):
            m = copy.deepcopy(self.m)
            by = {n["id"]: n for n in m["nodes"]}
            if label == "residual":
                by["final_result"]["inputs"][0] = "x"
            elif label == "gamma":
                by["ffn_normalized"]["inputs"][1] = "x"
            elif label == "norm":
                by["gate"]["inputs"][0] = "normalized"
            elif label == "gate":
                by["activation"]["inputs"][0] = "up"
            elif label == "output":
                next(
                    n
                    for n in m["nodes"]
                    if n["op"] == "output" and n["host"] == "new_key"
                )["host"] = "result"
            else:
                m["nodes"].append(dict(id="detached", op="input", inputs=[]))
            with self.subTest(label=label), self.assertRaises(Error):
                canonical(m)


class ParentRanges(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = parse(
            ROOT / "benchmarks/inference/waferllm/projected_cache_ffn_3x256x512x512_16x16/hls.cpp"
        )

    def test_parent_derived_boundary(self):
        from projected_cache_ffn_plan import plan

        s = plan(self.m)
        self.assertEqual(s["boundary"]["producer"], "result")
        self.assertEqual(s["attention"]["numerical_bounds"]["result"], 34.0625)
        self.assertEqual(sum(s["memory_per_pe"].values()), 44030)
        self.assertTrue(s["storage_lifetimes"]["validation"]["checked"])
        self.assertEqual(len(s["storage_lifetimes"]["phases"]), 23)
        self.assertEqual(s["resources"]["local_tasks"], [10, 11, 12, 14, 15, 16, 17])
        self.assertLessEqual(s["numerical_bounds"]["result"], 165)

    def test_parent_precision_and_host_rejections(self):
        from projected_cache_ffn_plan import plan

        for field in ("dtype", "host", "block", "statistic"):
            m = copy.deepcopy(self.m)
            by = {n["id"]: n for n in m["nodes"]}
            if field == "dtype":
                by["wu"]["dtype"] = "f32"
            elif field == "host":
                by["wu"]["host"] = "x"
            elif field == "block":
                by["up"]["block_size"] = 3
            else:
                by["ffn_normalized"]["dataflow"]["statistic"] = "sum"
            with self.subTest(field=field), self.assertRaises(Error):
                plan(m)
