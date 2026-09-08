"""Cyclic alignment and fail-closed resource planning for Cannon."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import copy, sys, unittest
from pathlib import Path
import numpy as np

ROOT = repository_root(__file__)
sys.path.insert(0, str(ROOT / "lib"))
from frontend import parse
from ir import verify
from mesh_cannon import pack, plan


class Cannon(unittest.TestCase):
    def test_skew_and_round_ownership(self):
        p, t = 4, 3
        n = p * t
        a = np.arange(n * n, dtype=np.float32).reshape(n, n)
        b = a + 1024
        ap, bp = pack(a, p, "A"), pack(b, p, "B")
        total = np.zeros((n, n), dtype=float)
        for step in range(p):
            for y in range(p):
                for x in range(p):
                    k = (x + y + step) % p
                    np.testing.assert_array_equal(
                        ap[y, x].reshape(t, t),
                        a[y * t : (y + 1) * t, k * t : (k + 1) * t],
                    )
                    np.testing.assert_array_equal(
                        bp[y, x].reshape(t, t),
                        b[k * t : (k + 1) * t, x * t : (x + 1) * t],
                    )
                    total[y * t : (y + 1) * t, x * t : (x + 1) * t] += ap[y, x].reshape(
                        t, t
                    ).astype(float) @ bp[y, x].reshape(t, t).astype(float)
            ap = np.roll(ap, -1, axis=1)
            bp = np.roll(bp, -1, axis=0)
        np.testing.assert_array_equal(total, a.astype(float) @ b.astype(float))

    def test_export_handles_survive_working_pointer_rotation(self):
        # Model the pointer assignments in the generated synchronous swap body.
        # SDK memcpy follows the exported cell's current value at the next upload.
        import re, tempfile
        from mesh_cannon import generate

        def after_call(code, rounds):
            pointers = dict(
                re.findall(r"var\s+(\w+)\s*:\s*\[\*\]f32\s*=\s*&(\w+)\s*;", code)
            )
            exports = {
                port: cell
                for cell, port in re.findall(
                    r'@export_symbol\((\w+),\s*"([AB])"\)', code
                )
            }
            begin = code.index("fn compute() void {")
            end = code.index("\ncomptime", begin)
            assignments = re.findall(
                r"(?:var\s+)?(\w+)\s*=\s*(\w+)\s*;", code[begin:end]
            )
            self.assertEqual(len(assignments), 6)
            for _ in range(rounds - 1):
                for target, origin in assignments:
                    pointers[target] = pointers[origin]
            return {port: pointers[cell] for port, cell in exports.items()}

        for p in (4, 8):
            source = (
                ROOT
                / f"benchmarks/linear_algebra/matrix_algorithms/mesh_cannon_{64 if p==4 else 128}_{p}x{p}_vector/hls.cpp"
            )
            schedule = plan(verify(parse(source), 4, 2))
            with tempfile.TemporaryDirectory() as td:
                generate(schedule, td)
                code = (Path(td) / "pe.csl").read_text()
            expected = {"A": "Matrix_1", "B": "Matrix_2"}
            self.assertEqual(after_call(code, p), expected)
            old = code.replace(
                '@export_symbol(host_A_ptr, "A")', '@export_symbol(A_ptr, "A")'
            ).replace('@export_symbol(host_B_ptr, "B")', '@export_symbol(B_ptr, "B")')
            if p == 4:
                self.assertEqual(
                    after_call(old, p), expected
                )  # why the smaller test hid the bug
            else:
                self.assertNotEqual(after_call(old, p), expected)

    def test_frontend_resources(self):
        raw = parse(
            ROOT / "benchmarks/linear_algebra/matrix_algorithms/mesh_cannon_64_4x4_vector/hls.cpp"
        )
        m = verify(raw, 4, 2)
        s = plan(m)
        self.assertEqual(s["profile"], "mesh_cannon.v1")
        self.assertEqual(s["resources"]["input_queues"], [2, 3])
        for p in (3, 5):
            bad = copy.deepcopy(raw)
            bad["nodes"][2]["dataflow"]["rows"] = p
            bad["nodes"][2]["dataflow"]["cols"] = p
            with self.assertRaises(ValueError):
                verify(bad, 4, 2)
        bad = copy.deepcopy(raw)
        bad["nodes"][2]["dataflow"]["initial_align"] = "device"
        with self.assertRaises(ValueError):
            verify(bad, 4, 2)


if __name__ == "__main__":
    unittest.main()
