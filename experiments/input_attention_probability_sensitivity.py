"""Sensitivity study from actual SDK observations; not a simulated target policy.

Isolates observed probability mass error from V/PV/O/Z precision. The alternate
renormalization uses NumPy f64 exp/reduction, not unexecuted CSL f32 semantics.
"""

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
from mesh_common import unpack_tiles
from input_attention_fixtures import child_inputs
from attention_tail_fixtures import reference
from probe_runtime import verify


def analyze(root, output):
    root, output = Path(root), Path(output)
    assert not output.exists()
    verify(root)
    r = json.loads((root / "results.json").read_text())
    assert r["success"] and json.loads((root / "selected-cases.json").read_text()) == [
        3
    ]
    row = r["cases"][0]
    b = json.loads((root / "logical-inputs.json").read_text())[3]

    def d(k):
        return unpack_tiles(
            np.array(row[k], np.uint16).view(np.float16), 8, 8, "F"
        ).astype(float)

    values = d("input_normalized") @ np.array(b["v_weight"]).reshape(64, 64)
    logits = d("attention_logits")
    e = np.exp((logits - logits.max(axis=1)[:, None]) * 0.125)
    _, child = child_inputs(64, 64, 1e-6, b)
    z = reference(64, 64, 256, 1e-6, 0.125, child)[4]
    policies = []
    for name, p in (
        ("observed SDK half probabilities", d("attention_probability")),
        ("f64 mathematical softmax on observed SDK logits", e / e.sum(axis=1)[:, None]),
    ):
        observed_z = p @ values @ np.array(b["output_weight"]).reshape(
            64, 64
        ) + np.array(b["input_x"]).reshape(64, 64)
        policies.append(
            dict(
                name=name,
                z_relative_l2=float(np.linalg.norm(observed_z - z) / np.linalg.norm(z)),
                max_probability_mass_error=float(np.max(np.abs(p.sum(axis=1) - 1))),
            )
        )
    result = dict(
        scope=__doc__,
        policies=policies,
        files={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                root / "results.json",
                root / "logical-inputs.json",
                root / "provenance.json",
            )
        },
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(analyze(*sys.argv[1:]))
