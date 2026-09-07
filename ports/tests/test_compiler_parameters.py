import sys, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "toolchain"))
from compiler_parameters import encode
from frontend import Error


class CompilerParameters(unittest.TestCase):
    def test_fail_closed_cli_boundary(self):
        self.assertEqual(
            encode({"n": "u16", "mode": "i16"}, {"n": 65535, "mode": -1}),
            "--params=n:65535,mode:-1",
        )
        for schema, values in [
            ({"n": "u16"}, {"n": -1}),
            ({"n": "u16"}, {"n": 65536}),
            ({"flag": "i16"}, {"flag": True}),
            ({"flag": "i16"}, {"flag": "false"}),
            ({"flag": "bool"}, {"flag": False}),
            ({"n": "u16"}, {"n": 1, "x": 2}),
            ({"n,x": "u16"}, {"n,x": 1}),
        ]:
            with self.assertRaises(Error):
                encode(schema, values)


if __name__ == "__main__":
    unittest.main()
