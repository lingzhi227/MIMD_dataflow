"""Read-only source comparison of generated31-node diagnostic CSL executions."""

import hashlib, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "toolchain")]
from input_attention_fixtures import check
from mesh_common import unpack_tiles
from probe_runtime import verify

PORTS = dict(
    input_normalized="input_normalized",
    input_q_raw="q_raw",
    input_k_raw="k_raw",
    attention_v="v_raw",
    x="q_rotated",
    attention_k="k_rotated",
    attention_logits="logits",
    attention_probability="probability",
    attention_snapshot="attention",
    projection_snapshot="projection",
    post_projection_z="post_projection_residual",
    normalized="normalized",
    down_snapshot="output",
    result="result",
    wide_accumulator="wide_accumulator",
    up_accumulator="up_accumulator",
    gate_accumulator="gate_accumulator",
)


def review(root, source, output):
    root, source, output = map(Path, (root, source, output))
    assert not output.exists(), "fresh report"
    verify(root)
    verify(source)
    r = json.loads((root / "results.json").read_text())
    sr = json.loads((source / "results.json").read_text())
    bs = json.loads((root / "logical-inputs.json").read_text())
    assert r["success"] and sr["success"] and len(r["cases"]) == len(sr["cases"]) == 3
    rows = []
    for epoch, (row, old, b) in enumerate(zip(r["cases"], sr["cases"], bs)):
        for name, source_name in PORTS.items():
            np.testing.assert_array_equal(row[name], old[source_name], err_msg=name)

        def decode(name):
            return unpack_tiles(
                np.asarray(row[name], np.uint16).view(np.float16), 8, 8, "F"
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
        numeric = check(
            64,
            64,
            256,
            1e-6,
            0.125,
            b,
            {"output": decode("result").ravel().tolist()},
            obs,
        )
        progress = np.asarray(row["input_prefix_progress"])
        wanted = [1, None, 8, 8, 8, 1, 1, 1, 1, 1, 1, epoch + 1]
        for index, value in enumerate(wanted):
            if value is not None:
                np.testing.assert_array_equal(
                    progress[:, :, index], np.full((8, 8), value)
                )
        offsets = [0, 1, 7, 2, 6, 3, 5, 4]
        np.testing.assert_array_equal(
            progress[:, :, 1], np.broadcast_to(np.array(offsets)[:, None], (8, 8))
        )

        def cycles(words):
            a = np.asarray(words, np.int64)
            a = a[:, :, 0] + (a[:, :, 1] << 16) + (a[:, :, 2] << 32)
            z = np.asarray(words, np.int64)
            end = z[:, :, 3] + (z[:, :, 4] << 16) + (z[:, :, 5] << 32)
            return int(np.max((end - a) % (1 << 48)))

        h, c = cycles(row["timing"]), cycles(old["timing"])
        rows.append(
            dict(
                epoch=epoch,
                numerical=numeric,
                exact_source_port_groups=len(PORTS),
                prefix_progress_passed=True,
                hls_cycles=h,
                source_cycles=c,
                ratio=h / c,
            )
        )
    report = dict(
        passed=True,
        scope="Three matched-source generated CSL diagnostic calls only. Eight-case native gate has a preserved cancellation failure; no31-node numerical qualification or hardware performance claim.",
        cases=rows,
        files={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                root / "provenance.json",
                root / "results.json",
                source / "provenance.json",
                source / "results.json",
            )
        },
        reviewer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    r = review(*sys.argv[1:])
    print("PASS", len(r["cases"]), "source-matched diagnostic calls")
