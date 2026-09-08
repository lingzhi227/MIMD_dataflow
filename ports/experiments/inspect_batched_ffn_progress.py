"""Read-only audit of saved FFN calls; never promotes partial process success."""

import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("bundle", type=Path)
p.add_argument("report", type=Path)
a = p.parse_args()
root = a.bundle.resolve()
assert not a.report.exists()
sys.path.insert(0, str(root / "implementation"))
from integrity import verify_bundle, verify_codegen
from mesh_batched_feed_forward_sdk import values, packed, extents, decode
from batched_ffn_reference import reference, original_math, accuracy

verify_bundle(root)
verify_codegen(root)
read = lambda n: json.loads((root / n).read_text())
s, m, batches, r = [
    read(n) for n in ("schedule.json", "semantic.json", "batches.json", "results.json")
]
assert (
    1 <= len(r["cases"]) == len(r["diagnostics"]) <= m["epochs"]
    and r["runtime_instances"] == 1
)
reports = []
for epoch, (batch, d, out) in enumerate(zip(batches, r["diagnostics"], r["cases"])):
    assert set(d) == set(extents(s))
    for name, n in extents(s).items():
        v = np.asarray(d[name])
        assert (
            v.shape == (s["P"], s["P"], n)
            and np.issubdtype(v.dtype, np.integer)
            and np.all((v >= 0) & (v < 65536))
        )
    raw, stages = reference(s, values(m, batch))
    raw.update(packed(s, m, batch))
    for name, v in raw.items():
        np.testing.assert_array_equal(
            d[name],
            np.asarray(v, np.float16).view(np.uint16),
            err_msg=f"{epoch} {name}",
        )
    np.testing.assert_array_equal(
        d["progress"], np.tile([1] * 7 + [epoch + 1], (s["P"], s["P"], 1))
    )
    assert np.all((np.asarray(d["queues"]) & 60) == 60) and out == decode(s, m, d)
    math = original_math(values(m, batch), s["epsilon"])
    checks = {k: accuracy(v, math[k]) for k, v in stages.items()}
    t = np.asarray(d["timing"], np.uint64)
    start = t[:, :, 0] + (t[:, :, 1] << 16) + (t[:, :, 2] << 32)
    end = t[:, :, 3] + (t[:, :, 4] << 16) + (t[:, :, 5] << 32)
    cycles = (end - start) & ((1 << 48) - 1)
    assert np.all((cycles > 0) & (cycles < 2**32))
    reports.append(dict(epoch=epoch, stages=checks, max_pe_cycles=int(cycles.max())))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
a.report.write_text(
    json.dumps(
        dict(
            passed=True,
            completed_calls=len(reports),
            expected_calls=m["epochs"],
            full_qualification=False,
            process_reported_success=r["success"],
            cases=reports,
            results_sha256=sha(root / "results.json"),
            manifest_sha256=sha(root / "manifest.json"),
            driver_sha256=sha(Path(__file__)),
            scope="Saved complete calls through frozen numerical implementation only; process completion and final qualification still required",
        ),
        indent=2,
    )
    + "\n"
)
print("PARTIAL AUDIT PASS", len(reports))
