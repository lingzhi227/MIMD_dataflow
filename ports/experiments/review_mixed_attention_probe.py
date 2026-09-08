"""Original-input fixed gates for experimental mixed-width actual SDK outputs."""

import hashlib, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "toolchain")]
from input_attention_fixtures import check
from mesh_common import unpack_tiles
from probe_runtime import verify


def review(root, output):
    root, output = Path(root), Path(output)
    assert not output.exists()
    verify(root)
    selected = json.loads((root / "selected-cases.json").read_text())
    r = json.loads((root / "results.json").read_text())
    assert r["success"] and len(r["cases"]) == len(selected)
    bs = json.loads((root / "logical-inputs.json").read_text())
    from mesh_twohop import cycle

    ring = cycle(8)
    rows = []
    for call, (idx, row) in enumerate(zip(selected, r["cases"])):

        def half(name):
            return unpack_tiles(
                np.asarray(row[name], np.uint16).view(np.float16), 8, 8, "F"
            ).astype(float)

        def wide(name):
            return unpack_tiles(
                np.asarray(row[name], np.uint32).view(np.float32), 8, 8, "F"
            ).astype(float)

        obs = {
            name: half(port)
            for name, port in dict(
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
                name: wide(port)
                for name, port in dict(
                    v_raw="mixed_v",
                    v="mixed_v",
                    probability="mixed_probability_snapshot",
                    attention="mixed_a",
                    projection="mixed_projection",
                ).items()
            }
        )
        numeric = check(
            64,
            64,
            256,
            1e-6,
            0.125,
            bs[idx],
            {"output": half("result").ravel().tolist()},
            obs,
        )
        prob = obs["probability"]
        score = obs["score"]
        e = np.exp((score - score.max(axis=1)[:, None]) * 0.125)
        expected = e / e.sum(axis=1)[:, None]
        mass = float(np.max(np.abs(prob.sum(axis=1) - 1)))
        rel = float(np.max(np.abs(prob - expected) / expected))
        assert mass <= 2e-6 and rel <= 2e-6, "f32 probability stage accuracy"
        z = wide("mixed_z")
        wanted = obs["projection"] + np.array(bs[idx]["input_x"]).reshape(64, 64)
        np.testing.assert_allclose(z, wanted, rtol=2e-7, atol=1e-12)
        norm = wide("mixed_normalized")
        g = np.array(bs[idx]["gamma"]).reshape(1, 64)
        ref = z * g / np.sqrt(np.mean(z * z, axis=1)[:, None] + 1e-6)
        error = float(np.linalg.norm(norm - ref) / max(np.linalg.norm(ref), 1e-30))
        assert error <= 2e-6, "f32 RMS stage accuracy"
        for y in range(8):
            for x in range(8):
                offset = (-ring.index(y)) % 8
                np.testing.assert_array_equal(
                    row["progress"][y][x], [offset, offset, 8, 8, 8, 1, call + 1, 3]
                )
                np.testing.assert_array_equal(
                    row["prelude_progress"][y][x], [offset, 8, 1, call + 1]
                )
                np.testing.assert_array_equal(
                    row["input_prefix_progress"][y][x],
                    [1, offset, 8, 8, 8, 1, 1, 1, 1, 1, 1, call + 1],
                )
                np.testing.assert_array_equal(
                    row["attention_progress"][y][x],
                    [(-ring.index(x)) % 8, offset, 8, call + 1],
                )
                np.testing.assert_array_equal(
                    row["score_progress"][y][x], [8, 1, call + 1]
                )
                np.testing.assert_array_equal(
                    row["score_roots"][y][x],
                    [ring[(ring.index(y) - k) % 8] for k in range(8)],
                )
        np.testing.assert_array_equal(
            row["rms_progress"], np.broadcast_to([1, call + 1], (8, 8, 2))
        )
        np.testing.assert_array_equal(
            row["attention_softmax_progress"], np.ones((8, 8, 6), np.uint16)
        )
        assert np.all((np.asarray(row["queues"]) & 248) == 248), "owned queues drained"
        rows.append(
            dict(
                case=idx,
                counters_roots_and_owned_queues_passed=True,
                numerical=numeric,
                probability_mass_error=mass,
                probability_max_relative_error=rel,
                resident_z_add_passed=True,
                observed_z_rms_relative_l2=error,
            )
        )
    report = dict(
        passed=True,
        scope="Experimental mixed-width CSL adapter executed in SDK2.10.1, original-eleven-input unchanged numerical gates. Not registered typed HLS lowering, full protocol/qualification or hardware performance.",
        cases=rows,
        files={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                root / "provenance.json",
                root / "results.json",
                root / "primitive-scope.json",
            )
        },
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    r = review(*sys.argv[1:])
    print("PASS", len(r["cases"]), "mixed SDK diagnostic cases")
