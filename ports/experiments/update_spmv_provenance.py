"""Regenerate reviewable SpMV source deltas without touching upstream files."""

from pathlib import Path
import difflib, hashlib, json

ROOT = Path(__file__).resolve().parents[1]
UP = ROOT / "projects/sdk_examples/upstream/benchmarks/spmv-hypersparse/src"
files = {
    "layout.csl": ("toolchain/runtime/spmv_layout.csl", "production"),
    "kernel.csl": ("toolchain/runtime/spmv_kernel.csl", "production"),
    "hypersparse_spmv/pe.csl": ("toolchain/runtime/spmv_pe.csl", "production"),
    "hypersparse_spmv/layout.csl": ("toolchain/runtime/spmv_routes.csl", "production"),
    "allreduce2R1E/pe.csl": (
        "experiments/reference/sdk-spmv-sync-migration/spmv_reduce_pe.csl",
        "unfinished auxiliary clock experiment; not linked",
    ),
    "allreduce2R1E/layout.csl": (
        "experiments/reference/sdk-spmv-sync-migration/spmv_reduce_routes.csl",
        "unfinished auxiliary clock experiment; not linked",
    ),
}
entries = {}
patch = []
for original, (target, role) in files.items():
    a, b = UP / original, ROOT / target
    entries[original] = dict(
        original_path=str(a.relative_to(ROOT)),
        target_path=target,
        role=role,
        original_sha256=hashlib.sha256(a.read_bytes()).hexdigest(),
        target_sha256=hashlib.sha256(b.read_bytes()).hexdigest(),
    )
    patch.extend(
        difflib.unified_diff(
            a.read_text().splitlines(True),
            b.read_text().splitlines(True),
            fromfile=str(a.relative_to(ROOT)),
            tofile=target,
        )
    )
report = dict(
    upstream_commit="4866cf330333446cb5e529e10f36be4600d1df29",
    sdk_sha256="fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d",
    changes=[
        "Flat CSL imports and explicit WSE3 queue initialization; RX_SOUTH moved off SDK-reserved input1.",
        "Four fixed-color TX queues and explicit send microthreads; conservative ownership, not evidence that all earlier resource hypotheses caused failure.",
        "Benchmark-only legacy clock allreduce is separated; production uses per-PE SDK timestamps with no global synchronization claim.",
        "Reverse-column scan guards occupied extent before metadata indexing, including unsigned empty/exhausted sentinel.",
        "Directional u32-aligned row staging preserves the logical u16 packet format and values; isolated SDK probes reproduced odd-offset body corruption before adaptation.",
        "Two sampled local partial sums and post-completion counts; full output correctness is checked independently from original CSC entries.",
    ],
    files=entries,
    added_library=dict(
        path="toolchain/runtime/u16_transport.csl",
        sha256=hashlib.sha256(
            (ROOT / "toolchain/runtime/u16_transport.csl").read_bytes()
        ).hexdigest(),
    ),
)
(ROOT / "docs/spmv-sdk2101-adapter.patch").write_text("".join(patch))
(ROOT / "docs/spmv-source-provenance.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
