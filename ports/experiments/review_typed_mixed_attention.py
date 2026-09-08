"""Fresh review snapshot: public native HLS, identical CSL, target and original-input gates."""

import datetime, hashlib, json, shutil, sys, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "toolchain")]
from mesh_input_attention_mixed import verify, plan, generate
from input_attention_mixed_audit import audit_cases
from mesh_feed_forward_sdk import decode
from mesh_common import unpack_tiles
from input_attention_fixtures import check
from probe_runtime import verify as verify_bundle


def review(bundle):
    bundle = Path(bundle)
    verify_bundle(bundle)
    assert (
        json.loads((bundle / "primitive-scope.json").read_text())["mixed_csl_adapter"]
        is False
    )
    native = json.loads((bundle / "native-gate.json").read_text())
    assert native["passed"] and len(native["checks"]) == 8
    raw = json.loads((bundle / "frontend.json").read_text())
    m = verify(raw, 8, 2)
    s = plan(m)
    bs = json.loads((bundle / "logical-inputs.json").read_text())
    r = json.loads((bundle / "results.json").read_text())
    rows = r["cases"]
    assert (
        r["success"]
        and len(rows) == 8
        and json.loads((bundle / "selected-cases.json").read_text()) == list(range(8))
    )
    output = (
        ROOT
        / "evidence"
        / (
            "input-attention-typed-review-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    output.mkdir()
    print(output.relative_to(ROOT), flush=True)
    shutil.copytree(
        ROOT / "toolchain",
        output / "implementation",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copyfile(__file__, output / "driver.py")
    for name in (
        "input_attention_fixtures.py",
        "attention_tail_fixtures.py",
        "prefill_tail_fixtures.py",
        "feed_forward_fixtures.py",
    ):
        shutil.copyfile(ROOT / name, output / name)
    with tempfile.TemporaryDirectory() as t:
        generate(s, t)
        files = list(Path(t).glob("*.csl"))
        assert len(files) == 12
        for f in files:
            assert f.read_bytes() == (bundle / f.name).read_bytes(), f.name
    (output / "current-checked-semantic.json").write_text(
        json.dumps(m, indent=2) + "\n"
    )
    (output / "current-checked-schedule.json").write_text(
        json.dumps(s, indent=2) + "\n"
    )
    result = dict(
        r,
        diagnostics=rows,
        cases=[decode(s, m, row) for row in rows],
        launches=["hls_main"] * 8,
    )
    target = audit_cases(s, m, bs, result)
    mathematics = []
    for b, row, final in zip(bs, rows, result["cases"]):

        def value(name, wide=False):
            return unpack_tiles(
                np.asarray(row[name], np.uint32 if wide else np.uint16).view(
                    np.float32 if wide else np.float16
                ),
                8,
                8,
                "F",
            ).astype(float)

        obs = {
            k: value(v)
            for k, v in dict(
                input_normalized="input_normalized",
                q_raw="input_q_raw",
                k_raw="input_k_raw",
                q="x",
                k="attention_k",
                score="attention_logits",
                delta="down_snapshot",
            ).items()
        }
        obs.update(
            {
                k: value(v, True)
                for k, v in dict(
                    v_raw="mixed_v",
                    v="mixed_v",
                    probability="mixed_probability_snapshot",
                    attention="mixed_a",
                    projection="mixed_projection",
                ).items()
            }
        )
        mathematics.append(check(64, 64, 256, s["epsilon"], s["scale"], b, final, obs))
    report = dict(
        passed=True,
        scope="Development typed HLS C++ -> checked IR/plan -> shared CSL -> actual SDK2.10.1 eight cases; current shared target audit plus independent original-eleven-input branch mathematics. Not generic public admission, catalog qualification or hardware/source-relative performance.",
        bundle=str(bundle),
        native_eight_passed=True,
        current_twelve_csl_files_exact=True,
        target=target,
        original_input_mathematics=mathematics,
    )
    (output / "review.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "provenance.json").write_text(
        json.dumps(
            dict(
                bundle_provenance_sha256=hashlib.sha256(
                    (bundle / "provenance.json").read_bytes()
                ).hexdigest(),
                results_sha256=hashlib.sha256(
                    (bundle / "results.json").read_bytes()
                ).hexdigest(),
                files={
                    str(p.relative_to(output)): hashlib.sha256(
                        p.read_bytes()
                    ).hexdigest()
                    for p in output.rglob("*")
                    if p.is_file()
                },
            ),
            indent=2,
        )
        + "\n"
    )
    print("PASS", len(rows), "typed HLS SDK cases")


if __name__ == "__main__":
    review(sys.argv[1])
