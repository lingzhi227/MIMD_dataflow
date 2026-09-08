"""Native diagnostic: can half-output blocked reductions cure residual cancellation?

These are explicit arithmetic experiments, not admitted device lowering policies.
No input, tolerance, baseline source or completed execution bundle is modified.
"""

import datetime, hashlib, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "toolchain")]
from native_transport import parse_outputs
from input_attention_fixtures import check


def run(baseline):
    baseline = Path(baseline).resolve()
    root = (
        ROOT
        / "evidence"
        / (
            "input-attention-precision-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    root.mkdir()
    print(root.relative_to(ROOT), flush=True)
    text = (baseline / "observed.cpp").read_text()
    bs = json.loads((baseline / "logical-inputs.json").read_text())
    indices = (
        "input_normalized",
        "q_raw",
        "k_raw",
        "v_raw",
        "q",
        "k",
        "score",
        "probability",
        "attention",
        "projection",
        "delta",
    )
    report = []
    for name, block in (("all-five-block8", 8), ("all-five-block1", 1)):
        dst = root / name
        dst.mkdir()
        source = text
        for a, b in (
            ("input_normalized", "q_weight"),
            ("input_normalized", "k_weight"),
            ("input_normalized", "v_weight"),
            ("probability", "v"),
            ("attention", "output_weight"),
        ):
            old = f"spatial::matmul({a},{b})"
            assert source.count(old) == 1
            source = source.replace(
                old,
                f"spatial::matmul_blocked<spatial::f16,spatial::scalar>({a},{b},{block})",
            )
        (dst / "source.cpp").write_text(source)
        cmd = json.loads((baseline / "observed-command.json").read_text())
        cmd[cmd.index(str(baseline / "observed.cpp"))] = str(dst / "source.cpp")
        cmd[-1] = str(dst / "native")
        (dst / "command.json").write_text(json.dumps(cmd) + "\n")
        r = subprocess.run(cmd, capture_output=True, text=True)
        (dst / "compile.log").write_text(r.stdout + r.stderr)
        assert r.returncode == 0
        r = subprocess.run(
            [cmd[-1]],
            input=(baseline / "native-input.txt").read_text(),
            capture_output=True,
            text=True,
        )
        (dst / "output.txt").write_text(r.stdout)
        (dst / "stderr.txt").write_text(r.stderr)
        assert r.returncode == 0
        cases = []
        for b, row in zip(bs, parse_outputs(r.stdout)):
            obs = {k: row["__observe_" + k] for k in indices}
            obs["v"] = obs["v_raw"]
            try:
                numeric = check(
                    64, 64, 256, 1e-6, 0.125, b, {"output": row["output"]}, obs
                )
                cases.append(dict(passed=True, numerical=numeric))
            except AssertionError as error:
                cases.append(dict(passed=False, error=repr(error)))
        report.append(
            dict(policy=name, passed=all(v["passed"] for v in cases), cases=cases)
        )
        print(name, [v["passed"] for v in cases], flush=True)
    (root / "review.json").write_text(
        json.dumps(
            dict(
                scope=__doc__, baseline=str(baseline.relative_to(ROOT)), policies=report
            ),
            indent=2,
        )
        + "\n"
    )
    (root / "driver.py").write_bytes(Path(__file__).read_bytes())
    (root / "baseline-hashes.json").write_text(
        json.dumps(
            {
                str(p.relative_to(baseline)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in baseline.rglob("*")
                if p.is_file()
                and (
                    p.suffix in (".hpp", ".cpp")
                    or p.name in ("native-input.txt", "logical-inputs.json")
                )
            },
            indent=2,
        )
        + "\n"
    )
    return root


if __name__ == "__main__":
    run(sys.argv[1])
