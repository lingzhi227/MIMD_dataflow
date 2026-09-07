"""Compare separately qualified FFT ownership policies without recomputing an FFT."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def read(root, name):
    return json.loads((root / name).read_text())


def audited(root):
    code = 'import sys,json;from pathlib import Path;p=Path(sys.argv[1]).resolve();sys.path.insert(0,str(p/"implementation"));from validate import audit;print(json.dumps(audit(p)))'
    return json.loads(
        subprocess.check_output([sys.executable, "-c", code, str(root)], text=True)
    )


def logical_bits(packed, schedule):
    """Direct integer permutation preserves every float bit, including signed zero."""
    p, t, n = (schedule[k] for k in ("rows", "T", "N"))
    physical = (
        np.asarray(packed, np.float32)
        .view(np.uint32)
        .reshape(p, p, n, t, t, 2)
        .transpose(0, 3, 1, 4, 2, 5)
        .reshape(n, n, n, 2)
    )
    return (
        physical
        if schedule.get("result_layout", "input_layout") == "input_layout"
        else physical.transpose(2, 0, 1, 3)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("restored", type=Path)
    parser.add_argument("transposed", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    roots = (args.restored, args.transposed)
    audits = [audited(root) for root in roots]
    assert all(a["passed"] for a in audits)
    schedules = [read(root, "schedule.json") for root in roots]
    r, t = schedules
    assert r.get("result_layout", "input_layout") == "input_layout"
    assert t["result_layout"] == "transposed_pencils"
    assert len(r["stages"]) == 7 and len(t["stages"]) == 5
    assert r["stages"][:5] == t["stages"]
    for key in (
        "profile",
        "rows",
        "cols",
        "N",
        "T",
        "local_length",
        "epochs",
        "transform",
        "input",
        "output",
        "instrumentation",
    ):
        assert r[key] == t[key], key
    for name in ("batches.json", "runtime-options.json"):
        assert read(roots[0], name) == read(roots[1], name), name
    assert (
        read(roots[0], "qualification.json")["sdk_sha256"]
        == read(roots[1], "qualification.json")["sdk_sha256"]
    )
    results = [read(root, "results.json") for root in roots]
    assert all(len(result["diagnostics"]) == r["epochs"] for result in results)
    rows = []
    for epoch in range(r["epochs"]):
        bits = [
            logical_bits(result["diagnostics"][epoch]["packed_output"], schedule)
            for result, schedule in zip(results, schedules)
        ]
        np.testing.assert_array_equal(bits[0], bits[1])
        rc, tc = [audit["cases"][epoch]["max_local_cycles"] for audit in audits]
        rows.append(
            dict(
                epoch=epoch,
                logical_device_bits_exact=True,
                restored_max_local_cycles=rc,
                transposed_max_local_cycles=tc,
                restored_over_transposed=rc / tc,
            )
        )
    args.output.write_text(
        json.dumps(
            dict(
                passed=True,
                new_sdk_execution=False,
                sources=[
                    dict(
                        bundle=str(root),
                        results_sha256=hashlib.sha256(
                            (root / "results.json").read_bytes()
                        ).hexdigest(),
                    )
                    for root in roots
                ],
                comparisons=rows,
                scope="Same mathematical transform, input, mesh, SDK options and instrumentation mode; distinct declared output ownership. Integer permutation only, no host FFT or rescaling. Five versus seven stages, including corresponding observation overhead. Maximum-local simulator intervals exclude host gathering and are not global or hardware latency.",
            ),
            indent=2,
        )
        + "\n"
    )
    print(args.output)


if __name__ == "__main__":
    main()
