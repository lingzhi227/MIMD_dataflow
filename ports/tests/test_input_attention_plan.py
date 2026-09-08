"""Structural/range/lifetime validation of the next unselected resident plan."""

import copy, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "toolchain"), str(ROOT / "experiments")]
from input_attention_source import source
from frontend import parse, Error
from mesh_input_attention import verify, plan
from region_lifetimes import verify as lifetimes


class InputAttentionPlan(unittest.TestCase):
    def module(self, m=64, f=256, mode="counters"):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "hls.cpp"
            p.write_text(source(m, 64, f))
            r = parse(p)
        r["instrumentation"] = mode
        return r

    def test_renamed_structural_composition(self):
        r = self.module()
        ids = {n["id"]: f"v{i}" for i, n in enumerate(r["nodes"])}
        for n in r["nodes"]:
            n["id"] = ids[n["id"]]
            n["inputs"] = [ids[x] for x in n["inputs"]]
            if n["op"] == "add":
                n["inputs"].reverse()
        s = plan(verify(r, 8, 2))
        self.assertEqual(len(s["storage_lifetimes"]["phases"]), 25)
        self.assertEqual(len(s["input_bindings"]), 11)
        self.assertEqual(s["input_bindings"]["input_x"], "residual")
        self.assertLess(s["input_prefix_bounds"]["q"]["pair"]["output_absolute"], 1)
        self.assertFalse(s["input_composition"]["intermediate_host_transfer"])

    def test_bounded_profiles_and_rejection(self):
        for m, f, mode in ((128, 256, "counters"), (64, 128, "sampled")):
            self.assertLessEqual(
                sum(
                    plan(verify(self.module(m, f, mode), 8, 2))[
                        "memory_per_pe"
                    ].values()
                ),
                49152,
            )
        with self.assertRaises(Error):
            verify(self.module(mode="sampled"), 8, 2)
        r = self.module()
        next(n for n in r["nodes"] if n.get("host") == "q_weight")["abs_bound"] = 0.125
        with self.assertRaises(Error):
            verify(r, 8, 2)

    def test_reject_unshared_activation_and_pair_convention(self):
        r = self.module()
        next(n for n in r["nodes"] if n["op"] == "rotate_pairs")[
            "pair_order"
        ] = "even_odd"
        with self.assertRaises(Error):
            verify(r, 8, 2)
        r = self.module()
        x = next(n for n in r["nodes"] if n.get("host") == "input_x")
        mm = next(n for n in r["nodes"] if n["op"] == "matmul")
        mm["inputs"][0] = x["id"]
        with self.assertRaises(Error):
            verify(r, 8, 2)

    def test_transport_matches_executed_source_and_preserves_inputs(self):
        import json, numpy as np
        from mesh_input_attention_sdk import packed, extents
        source = ROOT / "tests/fixtures/history/input-attention-source-20260907T122338297913Z"
        batches = json.loads((source / "logical-inputs.json").read_text())
        expected = json.loads((source / "inputs.json").read_text())
        m = verify(self.module(), 8, 2)
        s = plan(m)
        before = copy.deepcopy(batches)
        for b, row in zip(batches, expected):
            row["residual"] = row.pop("input_x")
            actual = packed(s, m, b)
            self.assertEqual(set(actual), set(s["input_bindings"].values()))
            for name, value in actual.items():
                np.testing.assert_array_equal(value, row[name])
                self.assertEqual(value.shape, (8,8,extents(s)[name]))
        self.assertEqual(before, batches)

    def test_private_sink_names_are_hygienic(self):
        r = self.module()
        for i, name in ((0, "__prefix_norm_sink"), (1, "__prefix_pair_sink")):
            old = r["nodes"][i]["id"]
            for node in r["nodes"]:
                node["inputs"] = [name if v == old else v for v in node["inputs"]]
            r["nodes"][i]["id"] = name
        self.assertEqual(verify(r, 8, 2)["profile"], "mesh_input_attention.v1")

    def test_composed_codegen_preserves_existing_attention(self):
        import json
        from mesh_attention_tail import generate as old_generate
        from mesh_input_attention import generate

        old = (
            ROOT
            / "tests/fixtures/history/run-20260907T114402035658Z"
        )
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td)
            old_generate(json.loads((old / "schedule.json").read_text()), dest)
            for p in dest.glob("*.csl"):
                self.assertEqual(p.read_bytes(), (old / p.name).read_bytes(), p.name)
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td)
            generate(plan(verify(self.module(), 8, 2)), dest)
            text = (dest / "pe.csl").read_text()
            self.assertNotIn("HLS_REGION_", text)
            # Prefix composition leases the same physical asynchronous tasks.
            self.assertEqual(text.count("@bind_local_task("), 4)
            self.assertIn("input_pair.apply(&x,&x", text)
            self.assertIn("rms_local.square_sum(&residual", text)
            self.assertTrue((dest / "pair_rotation_local.csl").is_file())

    def test_reject_early_pair_scratch_alias(self):
        r = copy.deepcopy(plan(verify(self.module(), 8, 2))["storage_lifetimes"])
        next(v for v in r["values"] if v["name"] == "Q_then_K_pair_scratch")[
            "first"
        ] = 6
        with self.assertRaises(Error):
            lifetimes(r["storage"], r["values"], r["phases"])


if __name__ == "__main__":
    unittest.main()
