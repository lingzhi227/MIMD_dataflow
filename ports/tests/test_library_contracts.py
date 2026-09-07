"""Native library matching must not silently relax HLS semantics or resources."""

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "toolchain"))
from frontend import parse, Error
from grid_ir import verify
from library_contracts import sdk_stencil


class LibraryContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = verify(
            parse(ROOT / "projects/sdk_examples/resident_stencil_2x2x32_t4/hls.cpp"),
            2,
            64,
        )

    def test_no_implicit_relaxation(self):
        with self.assertRaisesRegex(Error, "explicit FMA"):
            sdk_stencil(self.module, 8)

    def test_direction_and_microthread_contract(self):
        c = sdk_stencil(self.module, 8, allow_reassociation=True)
        self.assertEqual(c["coefficient_permutation"], [0, 1, 3, 2, 4, 5, 6])
        r = c["resources"]
        self.assertNotIn(r["send_microthread"], r["receive_queues_and_microthreads"])
        self.assertEqual(c["memory_budget"]["neighbor_blocks"], 4 * 8 * 4)

    def test_block_bounds(self):
        for block in (True, 0, 1, 33):
            with self.assertRaisesRegex(Error, "block size"):
                sdk_stencil(self.module, block, allow_reassociation=True)

    def test_transitive_benchmark_dependency(self):
        m = verify(
            parse(ROOT / "projects/sdk_examples/resident_stencil_1x1x7_t3/hls.cpp"),
            2,
            64,
        )
        with self.assertRaisesRegex(Error, "allreduce requiring width"):
            sdk_stencil(m, 2, allow_reassociation=True)

    def test_wrong_coefficient_is_not_a_stencil_match(self):
        m = copy.deepcopy(self.module)

        def change(node):
            if isinstance(node, list):
                if node[:3] == ["index", "b", ["const", "6", "int"]]:
                    node[2][1] = "5"
                for child in node:
                    change(child)

        change(m["nodes"][2]["body"]["body"])
        with self.assertRaisesRegex(Error, "matching coefficient"):
            sdk_stencil(m, 8, allow_reassociation=True)

    def test_large_history_rejected(self):
        m = verify(
            parse(ROOT / "projects/sdk_examples/resident_stencil_4x4x128_t16/hls.cpp"),
            2,
            64,
        )
        m["nodes"][2]["grid"]["steps"] = 128
        with self.assertRaisesRegex(Error, "memory budget"):
            sdk_stencil(m, 32, allow_reassociation=True)


if __name__ == "__main__":
    unittest.main()
