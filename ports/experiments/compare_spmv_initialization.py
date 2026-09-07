"""Compare preserved before/after initialization runs with identical inputs.

Audits each run through its own frozen implementation in a separate process.
Reports maximum PE-local intervals, not globally synchronized elapsed cycles.
"""

import argparse, hashlib, json, subprocess, sys, shutil, tempfile
from pathlib import Path
import numpy as np


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def maximum_cycles(diagnostic):
    t = np.asarray(diagnostic["timing"], dtype=np.uint64).reshape(-1, 6)
    a = t[:, 0] + (t[:, 1] << 16) + (t[:, 2] << 32)
    b = t[:, 3] + (t[:, 4] << 16) + (t[:, 5] << 32)
    return int(np.max((b - a) & ((1 << 48) - 1)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    assert not args.output.exists(), "comparison output must be fresh"
    paths = [args.before.resolve(), args.after.resolve()]
    # The historical auditor writes audit.json: run it on a disposable copy.
    # Preserved evidence remains immutable even when re-auditing old snapshots.
    with tempfile.TemporaryDirectory() as tmp:
        for index, p in enumerate(paths):
            copy = Path(tmp) / str(index)
            shutil.copytree(
                p,
                copy,
                ignore=shutil.ignore_patterns(
                    "simfab_traces", "out", "out.core", "wio_flows_tmpdir*"
                ),
            )
            subprocess.run(
                [sys.executable, str(copy / "implementation/validate.py"), str(copy)],
                check=True,
                capture_output=True,
            )
    assert read(paths[0] / "batches.json") == read(paths[1] / "batches.json")
    results = [read(p / "results.json") for p in paths]
    assert all(r["success"] for r in results)
    assert len(results[0]["cases"]) == len(results[1]["cases"])
    ports = [read(p / "semantic.json")["nodes"][-1]["host"] for p in paths]
    assert ports[0] == ports[1]
    epochs = []
    for i, (old, new) in enumerate(zip(results[0]["cases"], results[1]["cases"])):
        a = np.asarray(old[ports[0]], np.float32)
        b = np.asarray(new[ports[1]], np.float32)
        old_cycles, new_cycles = [maximum_cycles(r["diagnostics"][i]) for r in results]
        epochs.append(
            dict(
                epoch=i,
                before_max_local=old_cycles,
                after_max_local=new_cycles,
                after_over_before=new_cycles / old_cycles,
                changed_output_elements=int(
                    np.count_nonzero(a.view(np.uint32) != b.view(np.uint32))
                ),
                max_output_abs_difference=float(
                    np.max(np.abs(a.astype(float) - b.astype(float)))
                ),
            )
        )
    args.output.write_text(
        json.dumps(
            dict(
                scope="same-input maximum PE-local interval; independent frozen audits passed; no hardware/global-time claim",
                runs=[str(p) for p in paths],
                hashes=[
                    {
                        n: sha(p / n)
                        for n in ("results.json", "batches.json", "manifest.json")
                    }
                    for p in paths
                ],
                epochs=epochs,
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(epochs))


if __name__ == "__main__":
    main()
