"""Compare observed FFN stages and scoped intervals against a matched source control."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np
from probe_runtime import read, sha
from analyze_prefill_tail_source import review


def compare(hls, source):
    code = 'import json,sys;from pathlib import Path;p=Path(sys.argv[1]);sys.path.insert(0,str(p/"implementation"));from validate import audit;print(json.dumps(audit(p)))'
    audit = json.loads(
        subprocess.check_output(
            [sys.executable, "-c", code, str(hls.resolve())], text=True
        )
    )
    assert audit["passed"]
    source_review = review(source)
    s = read(hls / "schedule.json")
    g = read(source / "geometry.json")
    assert s["profile"] == "mesh_prefill_tail.v1"
    assert all(s[k] == g[k] for k in ("M", "N", "F", "P")) and s["epsilon"] == 1e-6
    upper = [v.get("accumulation") == "block_f32" for v in s["projection_stages"][:2]]
    source_upper = read(source / "provenance.json").get(
        "blocked_upper_accumulation", False
    )
    assert upper == [source_upper] * 2
    assert read(hls / "runtime-options.json") == read(source / "runtime-options.json")
    for option in ("--fabric-dims=", "--fabric-offsets="):
        a = [v for v in read(hls / "sdk-command.json") if v.startswith(option)]
        b = [v for v in read(source / "sdk-command.json") if v.startswith(option)]
        assert len(a) == 1 and a == b
    inputs = read(source / "logical-inputs.json")
    assert read(hls / "batches.json")[: len(inputs)] == inputs
    h, c = read(hls / "results.json"), read(source / "results.json")
    assert len(h["cases"]) == 8 and len(c["cases"]) == len(inputs) == 3
    rows = []
    for i, (sc, hc) in enumerate(zip(c["cases"], h["diagnostics"])):
        pairs = [
            ("projection", "projection_snapshot"),
            ("post_projection_residual", "post_projection_z"),
            ("result", "result"),
            ("normalized", "normalized"),
            ("output", "down_snapshot"),
            ("wide_accumulator", "wide_accumulator"),
        ]
        if source_upper:
            pairs += [
                ("up_accumulator", "up_accumulator"),
                ("gate_accumulator", "gate_accumulator"),
            ]
        for sk, hk in pairs:
            np.testing.assert_array_equal(sc[sk], hc[hk])
        sampled = s["instrumentation"] == "sampled"
        if sampled:
            np.testing.assert_array_equal(
                sc["up"], np.asarray(hc["up_history"])[:, :, -s["hidden_length"] :]
            )
        t = np.asarray(sc["timing"], np.int64)
        cycles = sum(
            (t[:, :, j + 3] - t[:, :, j]) * (1 << (16 * j)) for j in range(3)
        ) % (1 << 48)
        assert np.all((cycles > 0) & (cycles < 2**32))
        cc = int(cycles.max())
        ch = audit["cases"][i]["max_local_cycles"]
        rows.append(
            dict(
                matched_observed_ports=[a for a, b in pairs],
                target_bits_exact=True,
                sampled_up_half_observed=sampled,
                sampled_up_half_exact=True if sampled else None,
                hls_max_local_cycles=ch,
                source_max_local_cycles=cc,
                hls_to_source_ratio=ch / cc,
            )
        )
    return dict(
        passed=True,
        new_sdk_execution=False,
        blocked_upper_accumulation=source_upper,
        source_math_review=source_review,
        cases=rows,
        scope="Supplied attention-output projection, residual, normalized gated FFN with final postprojection Z residual. Same inputs, precision, fabric and SDK options; shared RMS and block accumulator libraries. HLS retains immutable public inputs; source inputs are destructive. Snapshot/copy differences remain in measured local WSE3 simulator intervals, especially sampled prefixes. Three matched source calls; HLS eight-call validation separate. Not full Prefill/Decode or hardware throughput.",
        hashes={
            str(p): sha(p)
            for p in (
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
    r = compare(a.hls, a.source)
    a.output.write_text(json.dumps(r, indent=2) + "\n")
    print(a.output, r["cases"])
