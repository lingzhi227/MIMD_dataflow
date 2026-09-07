"""Native harness must preserve target-produced f32 subnormals and failure diagnostics."""

import sys, tempfile, unittest, json, struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "toolchain"))
from compile import build
from native_transport import parse_outputs
from frontend import Error


class NativeTransport(unittest.TestCase):
    def test_decimal_output_recovers_actual_binary32_value(self):
        got = parse_outputs("epoch 0\nx 2 1.00000024 -0.499755859\n")
        self.assertEqual(got[0]["x"], [1.000000238418579, -0.499755859375])
        with self.assertRaises(Error):
            parse_outputs("epoch 0\nx 1 3.5e38\n")

    def test_output_protocol_is_typed_and_ordered(self):
        got = parse_outputs(
            "epoch 0\n@u32 code 2 16777217 4294967295\nx 2 -0 1.40129846e-45\nepoch 1\nx 1 2\n"
        )
        self.assertEqual(got[0]["code"], [16777217, 4294967295])
        self.assertEqual(
            struct.pack("<f", got[0]["x"][0]), struct.pack("<I", 0x80000000)
        )
        self.assertEqual(struct.pack("<f", got[0]["x"][1]), struct.pack("<I", 1))
        for bad in (
            "",
            "x 1 2",
            "epoch 1\nx 1 2",
            "epoch 0\nx 1 2\nx 1 3",
            "epoch 0\nx 2 1",
            "epoch 0\nx 1 nan",
            "epoch 0\n@u32 x 1 4294967296",
            "epoch 0\n@u32 x 1 -1",
            "epoch 0\nx 1 2\nepoch 1",
        ):
            with self.assertRaises(Error):
                parse_outputs(bad)

    def test_subnormal_input_bits(self):
        values = [
            struct.unpack("<f", struct.pack("<I", w))[0]
            for w in [1, 0x80000001, 0x007FFFFF, 0x00800000]
        ]
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "test.cpp"
            src.write_text(
                '#include "spatial.hpp"\nvoid design(){auto x=spatial::input<1,4>("x");spatial::output("result",x);}\n'
            )
            dest = Path(td) / "build"
            build(src, dest, epochs=1, bound=1, batches=[{"x": values}])
            line = next(
                v
                for v in (dest / "native-output.txt").read_text().splitlines()
                if v.startswith("result ")
            )
            got = list(map(float, line.split()[2:]))
            self.assertEqual(
                [struct.pack("<f", v) for v in values],
                [struct.pack("<f", v) for v in got],
            )
            self.assertEqual((dest / "native-stderr.txt").read_text(), "")
            self.assertIn(
                "-DMW_MAX_INPUT=4",
                json.loads((dest / "native-command.json").read_text()),
            )


if __name__ == "__main__":
    unittest.main()
