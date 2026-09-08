"""Preserve a bounded source probe and prepare an identical longer full run."""

import datetime, hashlib, json, shutil, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "toolchain"), str(ROOT / "experiments")]
from integrity import verify_bundle, verify_codegen
from projected_cache_ffn_transport import packed
from probe_runtime import verify, read, sha

old = ROOT / "evidence/hls-composed-source-20260908T022937908586Z"
bundle = (
    ROOT
    / "projects/waferllm/projected_cache_ffn_3x256x512x512_16x16/run-20260908T030001167477Z"
)
verify(old)
verify_bundle(bundle)
verify_codegen(bundle)
s = read(bundle / "schedule.json")
batches = read(bundle / "batches.json")
physical = read(old / "inputs.json")
assert len(batches) == len(physical) == 8
for batch, want in zip(batches, physical):
    actual = packed(s, batch)
    assert set(actual) == set(want)
    for k, v in actual.items():
        np.testing.assert_array_equal(v, np.asarray(want[k]), err_msg=k)
for p in bundle.glob("*.csl"):
    expected = p.read_text()
    if p.name == "batched_matmul_blocked.csl":
        expected = expected.replace('"batched_matmul_local.csl"', '"source_vecmat.csl"')
    assert (old / p.name).read_text() == expected, p.name
root = (
    ROOT
    / "evidence"
    / (
        "hls-composed-source-full8-"
        + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
)
root.mkdir()
parent = read(old / "provenance.json")
for name in parent["files"]:
    shutil.copy2(old / name, root / name)
shutil.copy2(ROOT / "experiments/composed_ffn/source_control.py", root / "driver.py")
shutil.copy2(__file__, root / "retry-preparation.py")
shutil.copy2(old / "provenance.json", root / "parent-source-provenance.json")
shutil.copy2(bundle / "manifest.json", root / "hls-manifest.json")
(root / "provenance.json").write_text(
    json.dumps(
        dict(
            scope="Fresh 8-call source-compute control with21600s budget after measured783.8s/call. Identical inputs/CSL to bounded probe; same full graph as normal compiler bundle except two local-kernel imports. Partial parent carries no full qualification.",
            parent_source_provenance_sha256=sha(old / "provenance.json"),
            hls_manifest_sha256=sha(bundle / "manifest.json"),
            original_functions_sha256=parent["original_functions_sha256"],
            upstream_sha256=parent["upstream_sha256"],
            files={p.name: sha(p) for p in root.iterdir() if p.is_file()},
        ),
        indent=2,
    )
    + "\n"
)
print(root.relative_to(ROOT), flush=True)
