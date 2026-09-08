"""Explicit resident attention+FFN storage candidates; not compiler admission."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "toolchain"))
from mean_statistic_bounds import plan_mean_statistic
from sdk2101_resources import check_default_memcpy


def candidate(p):
    b, n, seq, f, pad = 3, 256, 512, 512, 4
    nt, st, ft = n // p, seq // p, f // p
    cap = max(3 * b * nt, b * st, 2 * b * ft, pad)
    # Retain every exported observation until readback: no undocumented aliases.
    a = dict(
        X=2 * b * nt,
        Q=2 * b * nt,
        K=2 * nt * st,
        V=2 * nt * st,
        W=2 * nt * nt,
        score=2 * b * st,
        scaled=2 * b * st,
        exponents=2 * b * st,
        probability=2 * b * st,
        local_max=2 * pad,
        maximum=2 * pad,
        local_sum=2 * pad,
        sums=2 * pad,
        context=2 * b * nt,
        delta=2 * b * nt,
        result=2 * b * nt,
        score_partial=2 * b * st,
        context_partial=2 * b * nt,
        delta_partial=2 * b * nt,
        collective_send=4 * cap,
        collective_reduced=4 * cap,
        max_send=4 * pad,
        max_gathered=4 * p * pad,
        gamma=2 * nt,
        qkv_weights=6 * nt * nt,
        cosine=nt,
        sine=nt,
        normalized=2 * b * nt,
        rms_scratch=2 * b * nt,
        rms_sums=2 * pad,
        rms_history=4 * pad,
        projections=6 * b * nt,
        projection_partial=6 * b * nt,
        rotated_key=2 * b * nt,
        pair_scratch=4 * nt,
        query_pair_history=4 * b * nt,
        key_pair_history=4 * b * nt,
    )
    blocks = dict(
        score=(nt, st, min(32, nt)),
        value=(st, nt, min(32, st)),
        output=(nt, nt, min(32, nt)),
        qkv_query=(nt, nt, 4),
        qkv_key=(nt, nt, min(32, nt)),
        qkv_new_value=(nt, nt, min(32, nt)),
    )
    for name, (inner, columns, block) in blocks.items():
        a[name + "_block_partial"] = 2 * (columns if block < inner else 1)
        a[name + "_block_total"] = 4 * (columns if block < inner else 1)
    extra = dict(
        ffn_weights=6 * nt * ft,
        ffn_normalized=2 * b * nt,
        ffn_square_scratch=2 * b * nt,
        ffn_sums=2 * pad,
        ffn_projections=4 * b * ft,
        ffn_activation=2 * b * ft,
        ffn_hidden=2 * b * ft,
        ffn_delta=2 * b * nt,
        ffn_result=2 * b * nt,
        ffn_rms_history=4 * pad,
        ffn_projection_partial=4 * b * ft,
        ffn_down_partial=2 * b * nt,
        mean_send=4 * pad,
        mean_reduced=4 * pad,
    )
    # Upper/lower blocked half -> f32 local partials; conservative distinct storage.
    for name, width in (("up", ft), ("gate", ft), ("down", nt)):
        extra["ffn_" + name + "_block_partial"] = 2 * width
        extra["ffn_" + name + "_block_total"] = 4 * width
    reserves = dict(
        protocol_descriptors=1024,
        prior_code_stack_reserve=20480,
        extra_ffn_code_stack_reserve=4096,
    )
    total = sum(a.values()) + sum(extra.values()) + sum(reserves.values())
    resources = dict(
        colors=[0, 1, 4, 5],
        input_queues=[2, 3, 4, 5],
        output_queues=[2, 3, 4, 5],
        local_tasks=[10, 11, 12, 14, 15, 16, 17],
    )
    check_default_memcpy(resources)
    return dict(
        P=p,
        numeric_attention=a,
        numeric_ffn=extra,
        reserves=reserves,
        proposed_total_bytes=total,
        headroom_bytes=49152 - total,
        collective_capacity=cap,
        resources=resources,
        next_rms_range=plan_mean_statistic(34.5625, nt, p, n, pad),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("report", type=Path)
    a = p.parse_args()
    assert not a.report.exists()
    plans = [candidate(8), candidate(16)]
    # P8 retains the qualified inventory, except deliberately enlarged common buffers.
    s = (
        ROOT
        / "projects/waferllm/projected_cache_attention_3x256x512_8x8/run-20260907T235739454777Z/schedule.json"
    )
    old = json.loads(s.read_text())["numeric_allocations"]
    for k, v in old.items():
        assert plans[0]["numeric_attention"][k] == (
            4 * 384 if k in ("collective_send", "collective_reduced") else v
        )
    assert set(old) == set(plans[0]["numeric_attention"])
    assert plans[0]["headroom_bytes"] < 0 < plans[1]["headroom_bytes"]
    files = [
        Path(__file__),
        s,
        ROOT / "toolchain/mean_statistic_bounds.py",
        ROOT / "toolchain/sdk2101_resources.py",
    ]
    r = dict(
        scope="Analysis-only allocation candidates. P16 requires new lowering, numerical proofs, SDK acceptance, ELF/stack and cycle measurement. No automatic rescaling of P8 qualification.",
        plans=plans,
        residual="FFN adds to resident attention result Z; reuses source-shared gamma.",
        blockers=[
            "P16 MAX/SUM, region routing and all contractions require explicit qualification.",
            "Mean range proof does not supply normalized L1, FFN nonlinear or output accuracy proof.",
            "4096 additional code/stack reserve is an engineering allowance, not measured sufficiency.",
            "Every activation and observation is distinct; future reuse needs proven lifetimes.",
        ],
        files={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
    )
    a.report.write_text(json.dumps(r, indent=2) + "\n")
    print([(v["P"], v["proposed_total_bytes"], v["headroom_bytes"]) for v in plans])
