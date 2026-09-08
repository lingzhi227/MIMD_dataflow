"""Preserve labeled actual SDK numerical failure; this is not a qualification."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import hashlib, json, sys
from pathlib import Path
import numpy as np

ROOT = repository_root(__file__)
sys.path[:0] = [str(ROOT), str(ROOT / "lib")]
from input_attention_fixtures import child_inputs, check
from attention_tail_fixtures import reference
from mesh_common import unpack_tiles
from probe_runtime import verify


def review(root, output):
    root, output = Path(root), Path(output)
    assert not output.exists()
    verify(root)
    assert json.loads((root / "selected-cases.json").read_text()) == [3]
    r = json.loads((root / "results.json").read_text())
    assert r["success"] and len(r["cases"]) == 1
    b = json.loads((root / "logical-inputs.json").read_text())[3]
    row = r["cases"][0]

    def decode(port):
        return unpack_tiles(
            np.asarray(row[port], np.uint16).view(np.float16), 8, 8, "F"
        ).astype(float)

    obs = {
        name: decode(port)
        for name, port in dict(
            input_normalized="input_normalized",
            q_raw="input_q_raw",
            k_raw="input_k_raw",
            v_raw="attention_v",
            q="x",
            k="attention_k",
            v="attention_v",
            score="attention_logits",
            probability="attention_probability",
            attention="attention_snapshot",
            projection="projection_snapshot",
            delta="down_snapshot",
        ).items()
    }
    _, child = child_inputs(64, 64, 1e-6, b)
    expected = reference(64, 64, 256, 1e-6, 0.125, child)
    actual = {**obs, "z": decode("post_projection_z"), "output": decode("result")}
    branches = {}
    for key, wanted in zip(
        ("score", "probability", "attention", "projection", "z", "delta", "output"),
        expected,
    ):
        v = actual[key]
        err = v - wanted
        branches[key] = dict(
            relative_l2=float(np.linalg.norm(err) / max(np.linalg.norm(wanted), 1e-30)),
            peak_scaled_error=float(
                np.max(np.abs(err)) / max(np.max(np.abs(wanted)), 1e-30)
            ),
            expected_range=[float(wanted.min()), float(wanted.max())],
            actual_range=[float(v.min()), float(v.max())],
        )
    error = None
    try:
        check(
            64,
            64,
            256,
            1e-6,
            0.125,
            b,
            {"output": actual["output"].ravel().tolist()},
            obs,
        )
    except AssertionError as e:
        error = repr(e)
    assert error is not None, "update diagnosis if device now passes unchanged gate"
    report = dict(
        sdk_execution_completed=True,
        numerical_gate_passed=False,
        gate_error=error,
        case=3,
        branches=branches,
        scope="Actual generated31-node CSL SDK2.10.1 case3, fixed original-eleven-input mathematical contract. Expected diagnostic failure retained, not a qualified application.",
        files={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                root / "provenance.json",
                root / "results.json",
                root / "native-gate.json",
            )
        },
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    r = review(*sys.argv[1:])
    print("SDK completed; numerical gate:", r["numerical_gate_passed"])
    print(r["branches"])
