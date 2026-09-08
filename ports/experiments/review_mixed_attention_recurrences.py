"""Audit executed mixed contractions and half-block MLP at observed producer joins.

Original-eleven-input independent mathematical gates are a separate requirement.
Observed joins isolate recurrence/transport faults, not end-to-end accuracy.
"""

import copy, json, sys, hashlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "toolchain"), str(ROOT)]
from mesh_common import unpack_tiles
from mixed_matmul_reference import evaluate
from input_attention_precision_structure import canonical
from mesh_input_attention import tail as tail31
from mesh_attention_tail import tail as tail23
from mesh_prefill_tail import core as core17
from mesh_feed_forward import core as core13
from mesh_mlp import plan
from mesh_mlp_sdk import audit_cases, extents
from probe_runtime import verify


def review(root, output):
    root, output = Path(root), Path(output)
    assert not output.exists()
    verify(root)
    rows = json.loads((root / "results.json").read_text())["cases"]
    bs = json.loads((root / "logical-inputs.json").read_text())
    selected = json.loads((root / "selected-cases.json").read_text())
    raw = json.loads(
        (
            ROOT
            / "evidence/input-attention-mixed-frontend-20260907T134515612938Z/frontend.json"
        ).read_text()
    )
    raw["instrumentation"] = "counters"
    _, half = canonical(raw, len(rows), 2)
    mlp = core13(core17(tail23(tail31(half))))
    schedule = plan(mlp)
    diagnostics = []
    outputs = []
    batches = []
    contractions = []
    for idx, row in zip(selected, rows):

        def value(name, wide=False):
            return unpack_tiles(
                np.asarray(row[name], np.uint32 if wide else np.uint16).view(
                    np.float32 if wide else np.float16
                ),
                8,
                8,
                "F",
            ).astype(float)

        from mesh_rms import reference as rms_reference
        from projection_reference import project
        from mesh_pair_rotation import reference as pair_reference
        from mesh_score import reference as score_reference
        from mesh_input_attention import plan as prefix_plan

        ps = prefix_plan(half)
        x = np.asarray(bs[idx]["input_x"]).reshape(64, 64)
        gamma = np.asarray(bs[idx]["gamma"]).reshape(1, 64)
        hbits = lambda a: np.asarray(a, np.float16).view(np.uint16)
        fbits = lambda a: np.asarray(a, np.float32).view(np.uint32)
        expected_norm = rms_reference(
            dict(rows=8, cols=8, M=64, N=64, Mt=8, Nt=8, epsilon=1e-6), x, gamma
        )[-1]
        np.testing.assert_array_equal(
            hbits(value("input_normalized")), hbits(expected_norm)
        )
        cosine = np.asarray(bs[idx]["cosine"]).reshape(1, 32)
        sine = np.asarray(bs[idx]["sine"]).reshape(1, 32)
        for stage, weight, raw_port, pair_port in (
            (0, "q_weight", "input_q_raw", "x"),
            (1, "k_weight", "input_k_raw", "attention_k"),
        ):
            raw = project(
                ps["input_prefix_stages"][stage],
                value("input_normalized"),
                np.asarray(bs[idx][weight]).reshape(64, 64),
            )[0]
            np.testing.assert_array_equal(hbits(value(raw_port)), hbits(raw))
            rotated = pair_reference(value(raw_port), cosine, sine, "odd_even", True)[
                -1
            ]
            np.testing.assert_array_equal(hbits(value(pair_port)), hbits(rotated))
        expected_score = score_reference(
            dict(P=8, Mt=8, Nt=8), value("x"), value("attention_k")
        )[-1]
        np.testing.assert_array_equal(
            hbits(value("attention_logits")), hbits(expected_score)
        )
        np.testing.assert_array_equal(
            hbits(value("normalized")), hbits(value("mixed_normalized", True))
        )
        np.testing.assert_array_equal(
            fbits(value("mixed_z", True)),
            fbits(np.float32(value("mixed_projection", True)) + np.float32(x)),
        )
        np.testing.assert_array_equal(
            hbits(value("result")),
            hbits(
                np.float32(value("mixed_z", True)) + np.float32(value("down_snapshot"))
            ),
        )
        stages = [
            (
                "V",
                value("input_normalized"),
                np.asarray(bs[idx]["v_weight"]).reshape(64, 64),
                "mixed_v",
            ),
            (
                "PV",
                value("mixed_probability_snapshot", True),
                value("mixed_v", True),
                "mixed_a",
            ),
            (
                "O",
                value("mixed_a", True),
                np.asarray(bs[idx]["output_weight"]).reshape(64, 64),
                "mixed_projection",
            ),
        ]
        for name, a, b, port in stages:
            actual = value(port, True).astype(np.float32)
            expected = evaluate(a, b, 8)
            np.testing.assert_array_equal(
                actual.view(np.uint32),
                expected.view(np.uint32),
                err_msg=name + " source-order f32 FMA",
            )
        contractions.append(
            dict(
                case=idx,
                stages=["V", "PV", "O"],
                exact_f32_words=3 * 64 * 64,
                half_prefix_rms_qk_pair_score_exact=True,
                residual_and_narrowing_exact=True,
            )
        )
        d = {name: copy.deepcopy(row[name]) for name in extents(schedule)}
        d["x"] = copy.deepcopy(row["normalized"])
        d["result"] = copy.deepcopy(row["down_snapshot"])
        batch = {mlp["nodes"][0]["host"]: value("normalized").ravel().tolist()}
        batch.update({n["host"]: bs[idx][n["host"]] for n in mlp["nodes"][1:4]})
        batches.append(batch)
        diagnostics.append(d)
        outputs.append(
            {mlp["nodes"][-1]["host"]: value("down_snapshot").ravel().tolist()}
        )
    audit = audit_cases(
        schedule,
        mlp,
        batches,
        dict(
            success=True,
            runtime_instances=1,
            launches=["hls_main"] * len(rows),
            diagnostics=diagnostics,
            cases=outputs,
        ),
    )
    result = dict(
        passed=True,
        scope=__doc__,
        contractions=contractions,
        half_block_mlp=audit,
        results_sha256=hashlib.sha256((root / "results.json").read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    r = review(*sys.argv[1:])
    print("PASS", len(r["contractions"]), "mixed contraction/MLP recurrence audits")
