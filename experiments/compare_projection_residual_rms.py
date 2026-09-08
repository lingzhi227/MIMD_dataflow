"""Compare frozen generated CSL with executed, explicitly repaired source controls."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from probe_runtime import read, sha, verify
from analyze_projection_residual_rms import analyze


def compare(hls, source):
    code = 'import json,sys;from pathlib import Path;p=Path(sys.argv[1]);sys.path.insert(0,str(p/"implementation"));from validate import audit;print(json.dumps(audit(p)))'
    audit = json.loads(
        subprocess.check_output(
            [sys.executable, "-c", code, str(hls.resolve())], text=True
        )
    )
    assert audit["passed"]
    verify(source)
    source_review = analyze(source)
    assert source_review["mathematical_accuracy_passed"]
    repair = read(source / "provenance.json")["repair"]
    assert repair in ("both", "library")
    s = read(hls / "schedule.json")
    g = read(source / "geometry.json")
    assert all(s[k] == g[k] for k in ("M", "N", "P"))
    assert s["epsilon"] == 1e-6
    assert read(hls / "runtime-options.json") == read(source / "runtime-options.json")
    for option in ("--fabric-dims=", "--fabric-offsets="):
        a = [v for v in read(hls / "sdk-command.json") if v.startswith(option)]
        b = [v for v in read(source / "sdk-command.json") if v.startswith(option)]
        assert len(a) == 1 and a == b, "matched physical fabric required"
    inputs = read(source / "logical-inputs.json")
    assert read(hls / "batches.json")[: len(inputs)] == inputs
    h = read(hls / "results.json")
    c = read(source / "results.json")
    assert len(h["cases"]) == 8 and len(c["cases"]) == len(inputs) == 3
    rows = []
    for i, (sc, hc) in enumerate(zip(c["cases"], h["diagnostics"])):
        for key in ("result", "inverse"):
            np.testing.assert_array_equal(sc[key], hc[key])
        sampled = s["instrumentation"] == "sampled"
        if sampled:
            for key in ("sum", "local_square_sum", "reduced_square_sum"):
                np.testing.assert_array_equal(sc[key], hc[key])
            np.testing.assert_array_equal(
                sc["projection"], np.asarray(hc["history"])[:, :, -s["length"] :]
            )
        np.testing.assert_array_equal(sc["progress"], i + 1)
        t = np.asarray(sc["timing"], np.int64)
        cycles = sum(
            (t[:, :, j + 3] - t[:, :, j]) * (1 << (16 * j)) for j in range(3)
        ) % (1 << 48)
        assert np.all((cycles > 0) & (cycles < 2**32))
        cc = int(cycles.max())
        ch = audit["cases"][i]["max_local_cycles"]
        rows.append(
            dict(
                output_and_inverse_half_bits_exact=True,
                sampled_final_projection_and_norm_stages_exact=(
                    True if sampled else None
                ),
                sampled_final_projection_and_norm_stages_observed=sampled,
                hls_max_local_cycles=ch,
                source_max_local_cycles=cc,
                hls_to_source_ratio=ch / cc,
            )
        )
    return dict(
        passed=True,
        new_sdk_execution=False,
        source_repair=repair,
        shared_local_rms_library=repair == "library",
        source_math_review=source_review,
        cases=rows,
        scope="Supplied activation/weight/residual/gamma, same inputs, physical fabric and SDK options. Explicit source descriptor and row-inverse corrections. HLS preserves public inputs and reuses dead private buffers; source uses destructive inputs/separate arrays. Sampled HLS additionally records every projection prefix and initial operands. Both retain inverse observations; source local/reduced copies occur inside timed path. Unequal observation/copy costs remain in local WSE3 simulator interval ratios. No full Prefill or hardware throughput claim.",
        hashes={
            str(v): sha(v)
            for v in (
                hls / "manifest.json",
                hls / "results.json",
                source / "provenance.json",
                source / "results.json",
                Path(__file__),
            )
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("hls", type=Path)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    assert not a.output.exists()
    result = compare(a.hls, a.source)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(a.output, result["cases"])
