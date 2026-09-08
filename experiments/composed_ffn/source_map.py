"""Write the composed application's source traceability; never qualify a run."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import hashlib
import json
import re
from pathlib import Path

ROOT = repository_root(__file__)
PROJECT = ROOT / "benchmarks/inference/waferllm/projected_cache_ffn_3x256x512x512_16x16"


def main():
    source = ROOT / "third_party/sources/waferllm/Decode/src/decode.csl"
    digest = lambda data: hashlib.sha256(data).hexdigest()
    assert (
        digest(source.read_bytes())
        == "88921a433a9cf205ffdacbe7c5b0ff644d27788b70fdbdf1377b6d2984bdabeb"
    )
    text = source.read_text()
    prefix = json.loads(
        (
            ROOT
            / "benchmarks/inference/waferllm/projected_cache_attention_3x256x512_8x8/SOURCE-MAP.json"
        ).read_text()
    )
    rows = prefix["stages"]
    for row in rows:
        name = row["source_function"]
        if name == "xq_matvec_mult":
            row["mapping"] = (
                "Q local half block1/f32 merge; fused SDK QKV SUM. Block1 selected by unchanged 2% L2 / 3% peak original-input gates."
            )
        elif name in ("xk_matvec_mult", "xv_matvec_mult"):
            row["mapping"] = (
                "Local half block16/f32 merge; fused SDK QKV SUM. New rotated K and projected V are auxiliary outputs."
            )
        elif name == "score_matvec_mult":
            row["mapping"] = (
                "Read-only cache, feature-X contraction, half block16/f32 merge; source scalar factor represented by softmax scale."
            )
        elif name == "o_matvec_mult":
            row["mapping"] = (
                "Half block16/f32 merge; feature-X SDK SUM restores feature Y."
            )
        elif name == "attn_residual_add":
            row["hls_value"] = "attention_residual_Z"
            row["mapping"] = (
                "Actual X plus attention delta remains resident and feeds both FFN RMS and the final residual."
            )
    extra = [
        (
            "ffn_normalized",
            "rmsnorm_z",
            "Mathematical RMS uses original resident Z, shared gamma and epsilon. Repair the pinned source's squared-scratch numerator. Prescale local square sums in f32 before SDK SUM/broadcast and half narrowing; explicit statistic=mean.",
        ),
        (
            "up",
            "up_matvec_mult",
            "Y-input / X-hidden ownership, half block4/f32 local merge; fused UP/GATE SDK Y SUM.",
        ),
        (
            "gate",
            "gate_matvec_mult",
            "Same input and fused communication as UP; separate weights and disjoint output offset.",
        ),
        (
            "activation",
            "z2_silu",
            "Same SiLU equation, stable exp(-abs(x)) implementation using SDK half math; not literal source fast_exp arithmetic.",
        ),
        (
            "hidden",
            "z3_mat",
            "Local half UP times SiLU(GATE), with explicit value lifetime through DOWN.",
        ),
        (
            "ffn_delta",
            "down_matvec_mult",
            "X-hidden / Y-output ownership, half block4/f32 local merge and SDK X SUM.",
        ),
        (
            "final_result",
            "ffn_residual_add",
            "Original resident attention residual Z plus reduced FFN delta; one outer host completion after the full graph.",
        ),
    ]
    for value, name, mapping in extra:
        match = re.search(r"^fn " + name + r"\(", text, re.M)
        assert match, name
        start = match.start()
        pos = text.index("{", start) + 1
        depth = 1
        while depth:
            depth += (text[pos] == "{") - (text[pos] == "}")
            pos += 1
        rows.append(
            dict(
                hls_value=value,
                source_function=name,
                source_start_line=text[:start].count("\n") + 1,
                source_end_line=text[:pos].count("\n") + 1,
                function_sha256=digest(text[start:pos].encode()),
                mapping=mapping,
            )
        )
    report = dict(
        source=str(source.relative_to(ROOT)),
        source_commit=prefix["source_commit"],
        source_sha256=digest(source.read_bytes()),
        hls_sha256=digest((PROJECT / "hls.cpp").read_bytes()),
        generator_sha256=digest(Path(__file__).read_bytes()),
        stages=rows,
        scope="18 mathematical stages from the pinned migrated WSE3/SDK2.10.0 Decode tree. Not the paper's original WSE2 performance, full Decode, cache append, heads/GQA, masks or automatic position semantics. Source-compute control replaces only nine local contractions; RMS, pair transforms, softmax, SiLU and SDK communication are shared.",
    )
    destination = PROJECT / "SOURCE-MAP.json"
    assert not destination.exists(), "Preserve existing source maps"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
