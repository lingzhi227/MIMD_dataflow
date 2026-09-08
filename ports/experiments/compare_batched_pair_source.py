"""Full matched original Decode pair transform comparison, without model claims."""

import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np
from probe_runtime import read, sha, verify, SDK_HASH

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("hls", type=Path)
p.add_argument("source", type=Path)
p.add_argument("report", type=Path)
a = p.parse_args()
h, c = [v.resolve() for v in (a.hls, a.source)]
assert not a.report.exists()
code = 'import json,sys;from pathlib import Path;p=Path(sys.argv[1]);sys.path.insert(0,str(p/"implementation"));from validate import audit;print(json.dumps(audit(p)))'
aud = json.loads(
    subprocess.check_output([sys.executable, "-c", code, str(h)], text=True)
)
assert aud["passed"]
verify(c)
execution = read(c / "execution.json")
assert (
    execution["success"]
    and execution["sdk_sha256"] == SDK_HASH
    and execution["results_sha256"] == sha(c / "results.json")
)
q = read(h / "qualification.json")
assert q["success"] and q["sdk_sha256"] == SDK_HASH
s = read(h / "schedule.json")
assert s == read(c / "schedule.json") and s["layout"] == "batch_major"
for name in (
    "batches.json",
    "runtime-options.json",
    "semantic.json",
    "pe.csl",
    "layout.csl",
):
    assert (h / name).read_bytes() == (c / name).read_bytes()
origin = ROOT / "projects/waferllm/upstream/Decode/src/decode.csl"
text = origin.read_text()
original = text[text.index("fn xq_rope()") : text.index("fn xk_rope()")]
assert original == (c / "original-kernel.csl").read_text()
prov = read(c / "provenance.json")
assert prov["source_sha256"] == sha(origin) and prov["hls_manifest_sha256"] == sha(
    h / "manifest.json"
)
source_module = (c / "batched_pair_rotation_local.csl").read_text()
body = source_module[
    source_module.index("fn xq_rope()") : source_module.index("\nfn apply(")
]
a0 = body.index("        if(sampled!=0){")
a1 = body.index("        @load_to_dsr(dest_dsr_5, X_even_dsd);", a0)
restored = (body[:a0] + body[a1:]).replace(
    "\n    X_odd_dsd = @increment_dsd_offset(X_odd_dsd, 1, f16);", ""
)
assert restored.rstrip() == original.rstrip()
assert body.count("X_odd_dsd = @increment_dsd_offset(X_odd_dsd, 1, f16);") == 1
r, z = read(h / "results.json"), read(c / "results.json")
assert r["success"] and z["success"] and z["runtime_instances"] == 1
assert (
    len(r["cases"]) == len(z["cases"]) == len(z["diagnostics"]) == 6
    and z["launches"] == ["hls_main"] * 6
)
sys.path.insert(0, str(h / "implementation"))
from mesh_pair_rotation_sdk import extents, unpack

checks = []
for epoch, (hd, sd) in enumerate(zip(r["diagnostics"], z["diagnostics"])):
    for name, n in extents(s).items():
        raw = np.asarray(sd[name])
        assert (
            raw.shape == (s["rows"], s["cols"], n)
            and np.issubdtype(raw.dtype, np.integer)
            and np.all((raw >= 0) & (raw < 65536))
        )
        if name != "timing":
            np.testing.assert_array_equal(sd[name], hd[name])
    actual = (
        unpack(s, np.asarray(sd["result"], np.uint16).view(np.float16)).ravel().tolist()
    )
    assert z["cases"][epoch] == {"rotated": actual}
    t = np.asarray(sd["timing"], np.int64)
    cycles = sum((t[:, :, j + 3] - t[:, :, j]) * (1 << (16 * j)) for j in range(3)) % (
        1 << 48
    )
    assert np.all((cycles > 0) & (cycles < 2**32))
    hc = aud["cases"][epoch]["max_local_cycles"]
    sc = int(cycles.max())
    checks.append(
        dict(
            epoch=epoch,
            hls_max_pe_cycles=hc,
            source_max_pe_cycles=sc,
            ratio=hc / sc,
            exact_ports=[k for k in extents(s) if k != "timing"],
        )
    )
paths = (
    [h / n for n in ("manifest.json", "results.json", "qualification.json")]
    + [
        c / n
        for n in (
            "provenance.json",
            "results.json",
            "execution.json",
            "original-kernel.csl",
            "source-body-observation.diff",
        )
    ]
    + [Path(__file__), ROOT / "experiments/batched_pair_source_control.py"]
)
a.report.write_text(
    json.dumps(
        dict(
            passed=True,
            epochs=6,
            checks=checks,
            files={str(v.relative_to(ROOT)): sha(v) for v in paths},
            scope="Decode xq_rope with one explicit odd-offset repair after base reset and product observations. Source wrapper input preservation copy is inside timing; HLS writes separate output directly. Local simulator cycles only; not complete RoPE or inference.",
        ),
        indent=2,
    )
    + "\n"
)
print("BATCHED PAIR SOURCE COMPARISON PASS")
