"""Reviewable graph/range/storage candidate; does not claim codegen or SDK admission."""

import datetime, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "toolchain"))
from frontend import parse
from projected_cache_ir import canonical
from binary16 import quantize
from input_contracts import half_dot_bound, half_product_bound
from rms_l1_bounds import normalized_l1, projection
from pair_rotation_bounds import bound as pair_bound
from blocked_matmul import finite_bound
from positive_normalization_bounds import probability_mass, weighted_contraction
from ir import verify
from mesh_cache_attention import plan

p = ROOT / "projects/waferllm/projected_cache_attention_3x256x512_8x8/hls.cpp"
m = canonical(parse(p))
assert len(m["nodes"]) == 25
b, n, seq, pes = 3, 256, 512, 8
nt, st = n // pes, seq // pes
pad = (b + 1) // 2 * 2
total = quantize(pes * half_dot_bound(1, 1, nt))
l1 = normalized_l1(1, 1, total, n, nt, pes, 1e-6)
proj = projection(l1["bound"], 0.03125, nt, pes)
pair = pair_bound(proj["reduced"], 1, 1)
score_local = finite_bound(pair["output_absolute"], 1, nt, 32)
score = quantize(pes * score_local)
scaled = half_product_bound(score, 1 / 16)
mass = probability_mass(seq, st, pes)
ctx = weighted_contraction(seq, st, pes, 32, 1)
delta_local = finite_bound(ctx["reduced"], 0.125, nt, 32)
delta = quantize(pes * delta_local)
r = parse(ROOT / "projects/waferllm/cache_attention_5x256x512_8x8/hls.cpp")
for node in r["nodes"]:
    if node.get("shape") and node["shape"][0] == 5:
        node["shape"][0] = b
base = plan(verify(r, 8, 2))
alloc = dict(base["numeric_allocations"])
alloc.update(
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
    collective_send=4 * 3 * b * nt,
    collective_reduced=4 * 3 * b * nt,
)
memory = dict(alloc, protocol_descriptors=1024, code_stack_reserve=20480)
assert sum(memory.values()) <= 49152
phases = [
    "RMS local squares/sums",
    "SDK Y RMS SUM",
    "RMS normalize / local QKV",
    "SDK Y fused QKV SUM",
    "Q/K pairs / local score",
    "SDK X score SUM",
    "scale/local MAX",
    "SDK Y MAX gather/broadcast",
    "exp/local SUM",
    "SDK Y denominator SUM",
    "normalize/local PV",
    "SDK Y context SUM",
    "local output projection",
    "SDK X output SUM",
    "original residual / output completion",
]
report = dict(
    scope="Design candidate: typed structure and conservative range/storage analysis only; no synthesized CSL, native or SDK graph qualification yet",
    shape=dict(B=b, N=n, S=seq, P=pes),
    nodes=m,
    finite_ranges=dict(
        rms_sum=total,
        rms_l1=l1,
        projection=proj,
        rotated=pair,
        score_local=score_local,
        score=score,
        scaled=scaled,
        difference=quantize(2 * scaled),
        exponent=[0, 1],
        denominator=[1, seq],
        probability_mass=mass,
        context=ctx,
        delta_local=delta_local,
        delta=delta,
        result=quantize(delta + 1),
    ),
    numeric_allocations=alloc,
    memory_per_pe=memory,
    planned_bytes=sum(memory.values()),
    phases=phases,
    public_outputs=[
        "residual output feature Y",
        "new rotated key feature X",
        "new projected value feature X",
    ],
    cache_semantics="Supplied read-only cache shared across batch; new K/V do not enter attention or modify it",
    source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
)
root = (
    ROOT
    / "evidence"
    / (
        "projected-cache-contract-"
        + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
)
root.mkdir()
(root / "contract.json").write_text(json.dumps(report, indent=2) + "\n")
(root / "driver.py").write_bytes(Path(__file__).read_bytes())
print(root.relative_to(ROOT), report["planned_bytes"], proj, pair, ctx, delta)
