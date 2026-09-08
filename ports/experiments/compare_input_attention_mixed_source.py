"""Compare executed HLS composition to a precision-matched pinned source engine.

Both use shared f32 communication/math primitives; this measures composition
and observation overhead, not independent primitive implementations or hardware.
"""

import argparse, datetime, hashlib, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "toolchain")]
from probe_runtime import verify as source_verify, sha
from input_attention_fixtures import check
from mesh_common import unpack_tiles


def compare(bundle, source, output):
    bundle, source, output = map(Path, (bundle, source, output))
    assert not output.exists()
    source_verify(source)
    provenance = json.loads((source / "provenance.json").read_text())
    assert provenance["precision_matched_mixed"]
    # Bundle audit must execute through its frozen arithmetic implementation.
    import subprocess

    audit = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                "-c",
                'import json,sys;from pathlib import Path;p=Path(sys.argv[1]);sys.path.insert(0,str(p/"implementation"));from validate import audit;print(json.dumps(audit(p)))',
                str(bundle),
            ],
            text=True,
        )
    )
    assert audit["passed"]
    h = json.loads((bundle / "results.json").read_text())
    r = json.loads((source / "results.json").read_text())
    execution = json.loads((source / "execution.json").read_text())
    qualification = json.loads((bundle / "qualification.json").read_text())
    assert execution["success"] and qualification["success"]
    assert execution["results_sha256"] == sha(source / "results.json")
    assert execution["sdk_sha256"] == qualification["sdk_sha256"] == (
        "fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d")
    bs = json.loads((bundle / "batches.json").read_text())
    assert (
        h["success"]
        and r["success"]
        and h["runtime_instances"] == r["runtime_instances"] == 1
    )
    assert (
        bs == json.loads((source / "logical-inputs.json").read_text())
        and len(bs) == len(h["diagnostics"]) == len(r["cases"]) == 8
    )
    mappings = dict(
        input_normalized="input_normalized",
        q_raw="input_q_raw",
        k_raw="input_k_raw",
        q_rotated="x",
        k_rotated="attention_k",
        logits="attention_logits",
        normalized="normalized",
        output="down_snapshot",
        result="result",
        source_v="mixed_v",
        source_a="mixed_a",
        source_projection="mixed_projection",
        source_z="mixed_z",
        source_normalized="mixed_normalized",
        source_probability_snapshot="mixed_probability_snapshot",
        wide_accumulator="wide_accumulator",
        up_accumulator="up_accumulator",
        gate_accumulator="gate_accumulator",
        v_raw="attention_v",
        attention="attention_snapshot",
        projection="projection_snapshot",
        post_projection_residual="post_projection_z",
        probability="attention_probability",
    )
    checks = []
    ratios = []

    def cycles(row):
        t = np.asarray(row["timing"], np.int64)
        v = sum((t[:, :, j + 3] - t[:, :, j]) * (1 << (16 * j)) for j in range(3)) % (
            1 << 48
        )
        assert np.all(v > 0)
        return v

    for e, (b, hs, ss) in enumerate(zip(bs, h["diagnostics"], r["cases"])):
        for src, target in mappings.items():
            np.testing.assert_array_equal(
                ss[src], hs[target], err_msg=src + " matched source " + str(e)
            )

        def value(name, wide=False):
            return unpack_tiles(
                np.asarray(ss[name], np.uint32 if wide else np.uint16).view(
                    np.float32 if wide else np.float16
                ),
                8,
                8,
                "F",
            ).astype(float)

        obs = {
            name: value(port)
            for name, port in dict(
                input_normalized="input_normalized",
                q_raw="q_raw",
                k_raw="k_raw",
                q="q_rotated",
                k="k_rotated",
                score="logits",
                delta="output",
            ).items()
        }
        obs.update(
            {
                name: value(port, True)
                for name, port in dict(
                    v_raw="source_v",
                    v="source_v",
                    probability="source_probability_snapshot",
                    attention="source_a",
                    projection="source_projection",
                ).items()
            }
        )
        math = check(
            64,
            64,
            256,
            1e-6,
            0.125,
            b,
            {"output": value("result").ravel().tolist()},
            obs,
        )
        hc, sc = cycles(hs), cycles(ss)
        ratios.append(float(hc.max() / sc.max()))
        checks.append(
            dict(
                epoch=e,
                source_original_input_mathematics=math,
                exact_port_groups=len(mappings),
                hls_max_pe_cycles=int(hc.max()),
                source_max_pe_cycles=int(sc.max()),
                ratio_of_max_pe_cycles=ratios[-1],
                hls_cycles_per_pe=hc.tolist(),
                source_cycles_per_pe=sc.tolist(),
            )
        )
    dependencies = [
        bundle / "manifest.json",
        bundle / "results.json",
        bundle / "qualification.json",
        source / "provenance.json",
        source / "results.json",
        source / "execution.json",
        Path(__file__),
        ROOT / "input_attention_fixtures.py",
        ROOT / "attention_tail_fixtures.py",
        ROOT / "prefill_tail_fixtures.py",
        ROOT / "feed_forward_fixtures.py",
    ]
    report = dict(
        passed=True,
        scope=__doc__,
        source_mathematics_eight_passed=True,
        precision_matched=True,
        cases=checks,
        max_ratio=max(ratios),
        min_ratio=min(ratios),
        hashes={
            str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p): sha(p)
            for p in dependencies
        },
        timing_scope="WSE3 simulator per-PE cycle intervals; complete resident chain; includes observers, excludes host I/O; no hardware/full-model throughput claim.",
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("PASS source/HLS", len(checks), "max ratio", max(ratios))
    return report


if __name__ == "__main__":
    compare(*sys.argv[1:])
