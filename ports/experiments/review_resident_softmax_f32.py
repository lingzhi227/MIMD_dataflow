"""Audit executed f32 local/collective stages within the resident SDK probe."""

import hashlib, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "toolchain"), str(ROOT / "experiments")]
from mesh_common import unpack_tiles
from probe_runtime import verify


def review(root, output):
    root, output = Path(root), Path(output)
    assert not output.exists()
    verify(root)
    r = json.loads((root / "results.json").read_text())
    assert r["success"] and len(r["cases"]) == 1
    row = r["cases"][0]

    def words(name):
        return np.asarray(row[name], np.uint32).view(np.float32)

    logits = unpack_tiles(
        np.asarray(row["attention_logits"], np.uint16).view(np.float16), 8, 8, "F"
    ).astype(np.float32)
    probability = unpack_tiles(words("wide_probability"), 8, 8, "F")
    exponents = unpack_tiles(words("wide_exponents"), 8, 8, "F")
    history = words("wide_history").reshape(8, 8, 5, 8)
    scaled = np.float32(logits * 0.125)
    peaks = scaled.max(axis=1).reshape(8, 8)
    local = np.stack(
        [scaled[:, i * 8 : (i + 1) * 8].max(axis=1).reshape(8, 8) for i in range(8)],
        axis=1,
    )
    np.testing.assert_array_equal(history[:, :, 0, :], local)
    np.testing.assert_array_equal(
        history[:, :, 1, :], np.broadcast_to(peaks[:, None, :], (8, 8, 8))
    )
    ideal_exp = np.exp((scaled - scaled.max(axis=1)[:, None]).astype(float))
    exp_error = float(np.max(np.abs(exponents.astype(float) - ideal_exp) / ideal_exp))
    assert exp_error <= 2e-6
    sums = np.zeros((8, 8, 8), np.float32)
    for x in range(8):
        for k in range(8):
            sums[:, x, :] = np.float32(
                sums[:, x, :] + exponents[:, x * 8 + k].reshape(8, 8)
            )
    np.testing.assert_array_equal(history[:, :, 2, :], sums)
    left = sums[:, 0, :].copy()
    for x in range(1, 4):
        left = np.float32(left + sums[:, x, :])
    right = sums[:, -1, :].copy()
    for x in range(6, 4, -1):
        right = np.float32(right + sums[:, x, :])
    total = np.float32(np.float32(sums[:, 4, :] + right) + left)
    np.testing.assert_array_equal(
        history[:, :, 3, :], np.broadcast_to(total[:, None, :], (8, 8, 8))
    )
    inv = history[:, :, 4, :]
    inv_error = float(np.max(np.abs(inv.astype(float) * total[:, None, :] - 1)))
    assert inv_error <= 2e-6
    ideal = ideal_exp / ideal_exp.sum(axis=1)[:, None]
    error = float(np.max(np.abs(probability.astype(float) - ideal) / ideal))
    mass = float(np.max(np.abs(probability.astype(float).sum(axis=1) - 1)))
    assert error <= 2e-6 and mass <= 2e-6
    out = dict(
        passed=True,
        scope="Actual SDK2.10.1 f32 softmax primitive in resident31-node probe. Exact local/global maximum and f32 source-order sums; independent exp/reciprocal/probability/mass checks. Final chain still narrows probabilities and retains half V/PV/O/Z, not mixed pipeline qualification.",
        exp_max_relative_error=exp_error,
        reciprocal_max_relative_error=inv_error,
        probability_max_relative_error=error,
        probability_mass_error=mass,
        files={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (root / "provenance.json", root / "results.json")
        },
    )
    output.write_text(json.dumps(out, indent=2) + "\n")
    return out


if __name__ == "__main__":
    print(review(*sys.argv[1:]))
