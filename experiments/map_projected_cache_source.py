"""Generate a source traceability map from pinned Decode function bodies."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import hashlib, json, re
from pathlib import Path

ROOT = repository_root(__file__)
p = ROOT / "benchmarks/inference/waferllm/projected_cache_attention_3x256x512_8x8"
source = ROOT / "third_party/sources/waferllm/Decode/src/decode.csl"
text = source.read_text()
sha = lambda b: hashlib.sha256(b).hexdigest()
roles = [
    (
        "normalized",
        "rmsnorm_x",
        "Correct RMS mathematics over original X and gamma; reuse qualified repaired CSL RMS module and SDK SUM.",
    ),
    (
        "query",
        "xq_matvec_mult",
        "Q local contraction; explicit half block4/f32 merge and fused SDK QKV SUM.",
    ),
    (
        "new_key",
        "xk_matvec_mult",
        "K local contraction; half block32 and fused SDK QKV SUM.",
    ),
    (
        "new_value",
        "xv_matvec_mult",
        "V local contraction; half block32 and fused SDK QKV SUM; remains a public output.",
    ),
    (
        "rotated_query",
        "xq_rope",
        "Explicit odd_even four-product transform; known odd-offset reset defect repaired.",
    ),
    (
        "rotated_key",
        "xk_rope",
        "Same coefficient contract and repaired offset; remains a public output.",
    ),
    (
        "score",
        "score_matvec_mult",
        "Supplied old-cache contraction on feature X; source scalar factor represented by softmax scale.",
    ),
    (
        "probability",
        "softmax_score",
        "Stable original mathematical softmax; negative MAX seeding and row/DSD defects repaired; SDK half exp/reciprocal.",
    ),
    (
        "context",
        "output_matvec_mult",
        "Read-only supplied V cache, sequence-Y contraction and SDK SUM; no new-V append.",
    ),
    (
        "delta",
        "o_matvec_mult",
        "Output weight contraction, feature-X SDK SUM restores feature Y.",
    ),
    (
        "result",
        "attn_residual_add",
        "Original X plus output delta; exact original residual connectivity.",
    ),
]
rows = []
for hls, name, scope in roles:
    match = re.search(r"^fn " + name + r"\(", text, re.M)
    assert match, name
    start = match.start()
    pos = text.index("{", start) + 1
    depth = 1
    while depth:
        depth += (text[pos] == "{") - (text[pos] == "}")
        pos += 1
    body = text[start:pos]
    rows.append(
        dict(
            hls_value=hls,
            source_function=name,
            source_start_line=text[:start].count("\n") + 1,
            source_end_line=text[:pos].count("\n") + 1,
            function_sha256=sha(body.encode()),
            mapping=scope,
        )
    )
report = dict(
    source=str(source.relative_to(ROOT)),
    source_commit="fd1c2daae37cd68706c03fc8009887ecee9900f8",
    source_sha256=sha(source.read_bytes()),
    hls_sha256=sha((p / "hls.cpp").read_bytes()),
    stages=rows,
    scope="Traceability of mathematical stages and declared layout/arithmetic changes. This does not claim literal full Decode equivalence, cache append, mask, heads or independent source controls for shared modules.",
    generator_sha256=sha(Path(__file__).read_bytes()),
)
dest = p / "SOURCE-MAP.json"
assert not dest.exists()
dest.write_text(json.dumps(report, indent=2) + "\n")
item = json.loads((p / "PORT.json").read_text())
item["origins"] = ["Decode/src/decode.csl:" + v["source_function"] for v in rows]
item["source_map"] = "SOURCE-MAP.json"
(p / "PORT.json").write_text(json.dumps(item, indent=2) + "\n")
print(dest)
