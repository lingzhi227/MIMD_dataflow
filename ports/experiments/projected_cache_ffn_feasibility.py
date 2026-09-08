"""Source-backed next-boundary witness. Analysis only; never SDK qualification."""

import argparse, hashlib, json, sys, warnings
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("bundle", type=Path)
p.add_argument("report", type=Path)
a = p.parse_args()
assert not a.report.exists()
root = a.bundle.resolve()
sys.path.insert(0, str(root / "implementation"))
from integrity import verify_bundle
from mesh_projected_cache import values
from projected_cache_reference import reference, q
from sdk_axis_reference import half_sum

verify_bundle(root)
s, m = [json.loads((root / n).read_text()) for n in ("schedule.json", "semantic.json")]
b, n, seq, pes, nt = s["B"], s["N"], s["S"], s["P"], s["Nt"]
inputs = dict(
    x=[1.0] * (b * n),
    gamma=[1.0] * n,
    wq=[1 / 32] * (n * n),
    wk=[1 / 32] * (n * n),
    wv=[1 / 32] * (n * n),
    cosine=[1.0] * (n // 2),
    sine=[0.0] * (n // 2),
    key=[1.0] * (seq * n),
    value=[1.0] * (seq * n),
    wo=[1 / 8] * (n * n),
)
_, stages = reference(s, values(m, inputs))
z = stages["result"]
assert np.all(z == 33.0)
local = np.zeros((pes, s["padded_batches"]))
for y in range(pes):
    for j in range(nt):
        local[y, :b] = q(local[y, :b] + q(z[:, y * nt + j] ** 2))
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    reduced = half_sum(local, 0)
assert np.all(np.isfinite(local)) and np.all(np.isinf(reduced[:b]))
ffn_width = 512
new_weight_bytes = 2 * 3 * (n // pes) * (ffn_width // pes)
assert sum(s["memory_per_pe"].values()) + new_weight_bytes > 49152
repo = Path(__file__).resolve().parents[1]
source = repo / "projects/waferllm/upstream/Decode/src/decode.csl"
text = source.read_text()
assert "fn rmsnorm_z() void" in text and "fn ffn_residual_add() void" in text
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
report = dict(
    passed=True,
    sdk_executed=False,
    scope="Predicted legal-input boundary witness, not a ninth SDK case or new qualification. Existing attention graph remains finite; directly chaining the existing half-sum FFN RMS is not valid over its full declared producer range.",
    input_construction={
        k: dict(length=len(v), constant=v[0]) for k, v in inputs.items()
    },
    predicted_attention_output=33.0,
    next_rms_exact_sum=n * 33**2,
    next_rms_local_half_sums=local.tolist(),
    next_rms_global_half_overflow=True,
    numpy_warnings=[str(w.message) for w in caught],
    original_residual="FFN adds its delta to Z (attention output), not to original X.",
    source_gamma="Pinned rmsnorm_z and rmsnorm_x both load W_dsd; different gamma inputs are an extension, not inferred from this source.",
    memory=dict(
        current_sampled_plan=sum(s["memory_per_pe"].values()),
        additional_resident_ffn_weights_only=new_weight_bytes,
        lower_budget_before_new_code_and_activations=sum(s["memory_per_pe"].values())
        + new_weight_bytes,
        pe_limit=49152,
    ),
    required_design_work=[
        "Preserve a wider RMS statistic or scale to mean before the final half narrowing; derive bounds and validate actual SDK arithmetic before enabling.",
        "Re-plan resident weights and workspaces: larger PE mesh, explicitly communicating regions, or a separately justified lifetime/storage strategy. Simple concatenation is unsupported.",
        "Preserve Z as the final FFN residual and distinguish source-shared gamma from any future separate-gamma extension.",
    ],
    files={
        str(p): sha(p)
        for p in (
            root / "manifest.json",
            root / "source.cpp",
            root / "schedule.json",
            source,
            Path(__file__),
        )
    },
)
a.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
print("ANALYSIS ONLY: next RMS overflows; naive resident budget", report["memory"])
