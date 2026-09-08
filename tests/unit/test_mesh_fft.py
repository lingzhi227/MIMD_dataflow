
from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)

import sys, unittest, copy
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from frontend import parse, Error
from mesh_fft import (
    verify,
    plan,
    pack,
    unpack,
    twiddles,
    reference,
    phase_reference,
    phase_samples,
    logical_output,
)

ROOT = repository_root(__file__)


class FFTTests(unittest.TestCase):
    def module(self):
        return verify(
            parse(ROOT / "benchmarks/transforms/sdk_examples/fft3d_16_4x4_forward/hls.cpp"), 6, 8
        )

    def test_packing_source_offsets(self):
        m = self.module()
        s = plan(m)
        n = s["N"]
        t = s["T"]
        x = np.arange(n**3).reshape(n, n, n) * (1 + 2j)
        actual = pack(x, s)
        for y in range(n):
            for xx in range(n):
                for z in range(n):
                    o = 2 * (z * t * t + (y % t) * t + xx % t)
                    np.testing.assert_array_equal(
                        actual[y // t, xx // t, o : o + 2],
                        [x[y, xx, z].real, x[y, xx, z].imag],
                    )
        np.testing.assert_array_equal(unpack(actual, s), x)

    def test_packing_preserves_signed_zero_components(self):
        s = plan(self.module())
        n = s["N"]
        words = np.zeros((n, n, n, 2), np.uint32)
        words[..., 0] = 0x80000000
        x = words.view(np.complex64).reshape(n, n, n).astype(np.complex128)
        out = (
            unpack(pack(x, s), s)
            .astype(np.complex64)
            .view(np.uint32)
            .reshape(n, n, n, 2)
        )
        np.testing.assert_array_equal(out, words)

    def test_fail_closed(self):
        m = self.module()
        for field, value in [
            ("rows", 3),
            ("cols", 2),
            ("exchange", "trains"),
            ("result", "replicated"),
            ("fp", "strict"),
        ]:
            b = copy.deepcopy(m)
            b["nodes"][1]["dataflow"][field] = value
            with self.assertRaises(Error):
                plan(b)
        b = copy.deepcopy(m)
        b["nodes"][1]["fft"]["N"] = 128
        with self.assertRaises(Error):
            plan(b)

    def test_stage_permutations_restore_complete_transform(self):
        m = self.module()
        s = plan(m)
        n = s["N"]
        rng = np.random.default_rng(7)
        x = rng.normal(size=(n, n, n)) + 1j * rng.normal(size=(n, n, n))
        for direction in ("forward", "inverse"):
            for norm in ("backward", "ortho", "forward"):
                s["transform"].update(direction=direction, norm=norm)
                phases = phase_reference(x, s)
                np.testing.assert_allclose(
                    phases[-1], reference(x, s["transform"]), atol=1e-11
                )
                # The final two phases are pure invertible transposes, no FFT arithmetic.
                np.testing.assert_array_equal(phases[5], phases[4].transpose(2, 1, 0))
                np.testing.assert_array_equal(phases[6], phases[5].transpose(0, 2, 1))
        samples = phase_samples(x, s).reshape(4, 4, 16, 4)
        for py, px, local in ((0, 0, 0), (3, 2, 15), (1, 3, 7)):
            y = py * 4 + local // 4
            xx = px * 4 + local % 4
            np.testing.assert_array_equal(
                samples[py, px, local],
                np.float32(
                    [
                        x[y, xx, 0].real,
                        x[y, xx, 0].imag,
                        x[y, xx, -1].real,
                        x[y, xx, -1].imag,
                    ]
                ),
            )

    def test_transposed_output_is_explicit_storage_not_changed_transform(self):
        m = self.module()
        m["nodes"][1]["dataflow"]["result"] = "transposed_pencils"
        s = plan(m)
        self.assertEqual(s["phase_count"], 5)
        self.assertEqual(s["output_axes"], ["x", "z", "y"])
        n = s["N"]
        rng = np.random.default_rng(12)
        x = rng.normal(size=(n, n, n)) + 1j * rng.normal(size=(n, n, n))
        physical = phase_reference(x, s)[-1]
        np.testing.assert_allclose(
            logical_output(physical, s), reference(x, s["transform"]), atol=1e-11
        )
        self.assertFalse(np.allclose(physical, reference(x, s["transform"])))

    def test_large_spectral_input_bound_is_not_a_descriptor_extent(self):
        m = parse(ROOT / "benchmarks/transforms/sdk_examples/fft3d_16_4x4_forward/hls.cpp")
        checked = verify(m, 6, 262144)
        self.assertEqual(checked["input_bound"], 262144)
        with self.assertRaises(Error):
            verify(m, 6, 2147483648)

    def test_direction_normalization(self):
        n = 16
        x = np.ones((n, n, n), complex)
        for direction in ("forward", "inverse"):
            for norm in ("backward", "ortho", "forward"):
                r = reference(x, dict(direction=direction, norm=norm))
                scale = (
                    n**3
                    if (norm == "backward" and direction == "forward")
                    or (norm == "forward" and direction == "inverse")
                    else np.sqrt(n**3) if norm == "ortho" else 1
                )
                self.assertEqual(r[0, 0, 0], scale)
                self.assertEqual(np.count_nonzero(r), 1)
        self.assertGreater(twiddles(n)[3], 0)


if __name__ == "__main__":
    unittest.main()
