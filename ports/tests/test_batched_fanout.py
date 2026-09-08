"""Structural branch sharing, odd batches, target packing and resource checks."""

import copy, sys, unittest, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "toolchain"), str(ROOT)]
from frontend import parse, Error
from ir import verify
from mesh_batched_fanout import plan, inputs, reference
from mesh_batched_fanout_sdk import packed, decode
from batched_fanout_fixtures import batches, check
from grouped_collective_csl import generate


class BatchedFanout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = parse(ROOT / "projects/waferllm/batched_qkv_3x512x512_8x8_g2/hls.cpp")
        cls.m = verify(cls.raw, 8, 2)
        cls.s = plan(cls.m)

    def test_same_instance_extents_and_ownership(self):
        s = self.s
        self.assertEqual(s["resources"]["extent_transitions"], [4, 576])
        self.assertEqual(s["resources"]["colors"], [5, 6, 7, 8, 9])
        self.assertEqual([b["offset"] for b in s["branch_bindings"]], [0, 192, 384])
        self.assertLess(sum(s["memory_per_pe"].values()), 49152)
        with tempfile.TemporaryDirectory() as d:
            generate(d)
            text = (Path(d) / "axis_grouped_reduce_dynamic.csl").read_text()
            self.assertEqual(text.count("@set_dsd_length("), 11)
            self.assertIn("length<=@as(u16,bsz)", text)

    def test_ssa_reordering_and_public_port_renaming(self):
        from mesh_batched_fanout import evaluate, generate as emit

        altered = copy.deepcopy(self.raw)
        rename = {n["id"]: "renamed_" + n["id"] for n in altered["nodes"]}
        ports = {}
        for node in altered["nodes"]:
            node["id"] = rename[node["id"]]
            node["inputs"] = [rename[value] for value in node["inputs"]]
            if "host" in node:
                ports[node["host"]] = "public_" + node["host"]
                node["host"] = ports[node["host"]]
        altered["nodes"].reverse()
        other = verify(altered, 8, 2)
        other_schedule = plan(other)
        original_batches = batches(3, 512, 512, 3)
        renamed_batches = [
            {ports[k]: v for k, v in batch.items()} for batch in original_batches
        ]
        expected, _ = evaluate(self.m, original_batches)
        actual, _ = evaluate(other, renamed_batches)
        self.assertEqual(
            actual, [{ports[k]: v for k, v in row.items()} for row in expected]
        )
        with tempfile.TemporaryDirectory() as d:
            first, second = Path(d) / "first", Path(d) / "second"
            first.mkdir()
            second.mkdir()
            emit(self.s, first)
            emit(other_schedule, second)
            for source in first.glob("*.csl"):
                self.assertEqual(
                    source.read_bytes(), (second / source.name).read_bytes()
                )

    def test_shared_lifetime_checker_guards_observations_and_local_joins(self):
        from region_lifetimes import verify as check_lifetimes

        plan = self.s["storage_lifetimes"]
        self.assertTrue(plan["validation"]["checked"])
        self.assertEqual(plan["validation"]["phases"], 8)
        for name in ("X", "W", "weights", "result", "partial", "projections"):
            value = next(v for v in plan["values"] if v["name"] == name)
            self.assertEqual(value["last"], 7)
        changes = [
            lambda p: p["values"].append(
                dict(name="early_reuse", storage="result", bytes=2, first=3, last=3)
            ),
            lambda p: p["phases"][3].update(release=[]),
            lambda p: p["values"][0].update(last=3),
        ]
        for change in changes:
            bad = copy.deepcopy(plan)
            change(bad)
            with self.assertRaises(Error):
                check_lifetimes(bad["storage"], bad["values"], bad["phases"])

    def test_policy_alias_type_and_resource_rejection(self):
        base = self.m
        changes = [
            lambda m: m["nodes"][4]["dataflow"].update(fusion="none"),
            lambda m: m["nodes"][7]["inputs"].__setitem__(0, m["nodes"][0]["id"]),
            lambda m: m["nodes"][6].update(shape=[512, 256]),
            lambda m: m["nodes"][3].update(dtype="f32"),
            lambda m: m["nodes"][7]["dataflow"].update(groups=4),
        ]
        for change in changes:
            m = copy.deepcopy(base)
            change(m)
            with self.assertRaises((Error, KeyError)):
                verify(m, 8, 2)
        m = copy.deepcopy(base)
        for node in m["nodes"]:
            if node["op"] == "input" and node["shape"] == [512, 512]:
                node["shape"] = [512, 2048]
            elif node["op"] == "matmul":
                node["shape"] = [3, 2048]
        with self.assertRaisesRegex(Error, "memory"):
            verify(m, 8, 2)

    def test_all_branches_pack_decode_independent_math(self):
        s, m = self.s, self.m
        for batch in batches(3, 512, 512, 3):
            x, w, *weights = inputs(m, batch)
            *_, partial, total, outs = reference(s, x, w, weights)
            raw = {
                "projections": np.broadcast_to(
                    np.asarray(total, np.float16).view(np.uint16), (8, 8, 576)
                ).tolist()
            }
            logical = decode(s, m, raw)
            check(3, 512, 512, 3, batch, logical)
            for k, v in enumerate(outs):
                np.testing.assert_array_equal(logical["branch" + str(k)], v.ravel())
            transport = packed(s, m, batch)
            for y, col in ((0, 0), (7, 7), (2, 5)):
                for k, q in enumerate(weights):
                    np.testing.assert_array_equal(
                        transport["weights"][y, col, k * 4096 : (k + 1) * 4096],
                        q[y * 64 : (y + 1) * 64, col * 64 : (col + 1) * 64].ravel(),
                    )


class DecodeRouteAssociation(unittest.TestCase):
    def test_even_root_consumes_head_before_tail(self):
        from decode_grouped_reference import reduce
        from mesh_grouped_gemv import reduce_group

        signed = np.array([[2048.0], [0.0], [1.0], [-2048.0]])
        positive = np.array([[2048.0], [0.0], [1.0], [1.0]])
        self.assertEqual(float(reduce(signed)[0]), 0.0)
        self.assertEqual(float(reduce_group(signed, 2)[0]), 1.0)
        self.assertEqual(float(reduce(positive)[0]), 2048.0)
        self.assertEqual(float(reduce_group(positive, 2)[0]), 2050.0)


class BatchedRegisterLeases(unittest.TestCase):
    def test_declarations_match_bank_specific_inventory(self):
        import re
        from batched_resources import leases

        plan = leases(3)
        actual = set()
        for name in (
            "batched_rms_local.csl",
            "batched_matmul_local.csl",
            "axis_grouped_reduce.csl",
        ):
            text = (ROOT / "toolchain/runtime" / name).read_text()
            actual.update(
                (bank, int(index))
                for bank, index in re.findall(
                    r"@get_dsr\(dsr_(dest|src0|src1),\s*(\d+)\)", text
                )
            )
        planned = {
            (v["bank"], v["index"])
            for phase in plan["phases"]
            for v in phase["registers"]
        }
        self.assertEqual(actual, planned)
        self.assertEqual(
            {v["bank"] for v in plan["phases"][0]["registers"] if v["index"] == 2},
            {"dest", "src0"},
        )
        self.assertEqual(plan["phases"][1]["registers"], [dict(bank="src1", index=2)])


class SDKMemcpyReservations(unittest.TestCase):
    def test_data_plane_does_not_claim_sdk_control_plane(self):
        from sdk2101_resources import check_default_memcpy

        app = dict(
            colors=[5, 6, 7, 8, 9],
            input_queues=[3, 4, 5, 6, 7],
            output_queues=[3, 4, 5, 6, 7],
        )
        reserved = check_default_memcpy(app)
        self.assertEqual(reserved["input_queues"], [0, 1])
        for kind, identifier in (
            ("colors", 23),
            ("input_queues", 0),
            ("output_queues", 1),
            ("local_tasks", 21),
            ("local_tasks", 24),
            ("control_tasks", 33),
        ):
            value = copy.deepcopy(app)
            value[kind] = value.get(kind, []) + [identifier]
            with self.assertRaisesRegex(Error, "reserved"):
                check_default_memcpy(value)


if __name__ == "__main__":
    unittest.main()
