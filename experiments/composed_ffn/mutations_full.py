"""Reject corruption in immutable actual SDK snapshots with a sealed audit extension.

The numerical auditor is the bundle's frozen implementation. This driver adds
explicit launch-order and partial-success guards identified during review; it
does not modify the running bundle or pretend the guards were present there.
"""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bundle", type=Path)
    p.add_argument("snapshot", type=Path)
    p.add_argument("report", type=Path)
    a = p.parse_args()
    assert not a.report.exists()
    root = a.bundle.resolve()
    sys.path.insert(0, str(root / "implementation"))
    from integrity import verify_bundle, verify_codegen
    from projected_cache_ffn_reference import audit_cases

    verify_bundle(root)
    verify_codegen(root)
    s, m, batches = [
        json.loads((root / name).read_text())
        for name in ("schedule.json", "semantic.json", "batches.json")
    ]
    r = json.loads(a.snapshot.read_text())
    complete = r["success"]
    assert (
        complete is True and len(r["cases"]) == 8
    ), "Full-eight actual result required"

    def audit(value):
        count = len(value.get("diagnostics", []))
        assert value.get("launches") == ["hls_main"] * count, "outer launch sequence"
        assert not value.get("success") or count == len(batches), "premature success"
        return audit_cases(s, m, batches, value, require_complete=complete)

    baseline = audit(r)
    mutations = []
    for port in sorted(set(r["diagnostics"][0]) - {"timing", "queues"}):
        for y, x in ((0, 0), (15, 15)):

            def damage(v, port=port, y=y, x=x):
                v["diagnostics"][0][port][y][x][-1] ^= 1

            mutations.append((f"{port}-p{x}_{y}-word", damage))
    for port in ("X", "K", "V", "W", "ffn_weights", "ffn_rms_history", "ffn_result"):
        mutations.append(
            (
                port + "-truncated",
                lambda v, port=port: v["diagnostics"][0][port][0][0].pop(),
            )
        )
    for port in ("result", "new_key", "new_value"):
        mutations.append(
            (
                port + "-public",
                lambda v, port=port: v["cases"][0][port].__setitem__(
                    0, v["cases"][0][port][0] + 1
                ),
            )
        )
    mutations.extend(
        [
            ("launch-order", lambda v: v["launches"].__setitem__(0, "init_task")),
            ("missing-launch", lambda v: v["launches"].pop()),
            ("two-runtimes", lambda v: v.update(runtime_instances=2)),
            ("boolean-runtime", lambda v: v.update(runtime_instances=True)),
            ("integer-success", lambda v: v.update(success=1)),
            ("missing-output-call", lambda v: v["cases"].pop()),
            ("missing-raw-call", lambda v: v["diagnostics"].pop()),
            ("missing-port", lambda v: v["diagnostics"][0].pop("ffn_delta")),
            (
                "queue-not-empty",
                lambda v: v["diagnostics"][0]["queues"][15][15].__setitem__(0, 0),
            ),
            (
                "invalid-word",
                lambda v: v["diagnostics"][0]["ffn_delta"][15][15].__setitem__(
                    0, 65536
                ),
            ),
            (
                "zero-cycles",
                lambda v: v["diagnostics"][0]["timing"][15][15].__setitem__(
                    slice(3, 6), v["diagnostics"][0]["timing"][15][15][:3]
                ),
            ),
            (
                "invalid-timer",
                lambda v: v["diagnostics"][0]["timing"][15][15].__setitem__(5, 65535),
            ),
        ]
    )
    if not complete:
        mutations.append(("premature-success", lambda v: v.update(success=True)))
    else:
        mutations.append(("false-success", lambda v: v.update(success=False)))
    if len(r["cases"]) > 1:
        for port in (
            "K",
            "ffn_normalized",
            "ffn_projections",
            "ffn_hidden",
            "ffn_delta",
            "ffn_result",
        ):
            mutations.append(
                (
                    port + "-last-stale",
                    lambda v, port=port: v["diagnostics"][-1].__setitem__(
                        port, copy.deepcopy(v["diagnostics"][-2][port])
                    ),
                )
            )
    # Final-call coverage protects auxiliary K/V, immutable replicas and boundary
    # observations after all prior reentries, not only after initialization.
    for port in sorted(set(r["diagnostics"][-1]) - {"timing", "queues"}):

        def last_word(v, port=port):
            v["diagnostics"][-1][port][15][15][-1] ^= 1

        mutations.append((port + "-final-call-word", last_word))
    for port in ("result", "new_key", "new_value"):

        def last_public(v, port=port):
            v["cases"][-1][port][-1] += 1

        mutations.append((port + "-final-public", last_public))
    for port in (
        "rotated_key",
        "projections",
        "rms_history",
        "attention_progress",
        "ffn_rms_history",
    ):

        def stale(v, port=port):
            v["diagnostics"][-1][port] = copy.deepcopy(v["diagnostics"][-2][port])

        mutations.append((port + "-final-stale", stale))
    mutations.append(
        (
            "final-whole-call-replay",
            lambda v: v["diagnostics"].__setitem__(
                -1, copy.deepcopy(v["diagnostics"][-2])
            ),
        )
    )
    reports = []
    for name, damage in mutations:
        v = copy.deepcopy(r)
        damage(v)
        try:
            audit(v)
        except (ValueError, AssertionError, KeyError) as error:
            reports.append(dict(name=name, rejected=True, error=str(error)[:220]))
        else:
            raise AssertionError("accepted mutation: " + name)
        if len(reports) % 10 == 0:
            print("rejected", len(reports), flush=True)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    report = dict(
        passed=True,
        baseline=baseline,
        mutations=len(reports),
        cases=reports,
        manifest_sha256=sha(root / "manifest.json"),
        results_sha256=sha(a.snapshot),
        driver_sha256=sha(Path(__file__)),
        scope="Actual snapshot corruption through frozen numeric audit plus the explicitly recorded launch/partial-success extension. Partial snapshots are not full eight-call qualification.",
    )
    a.report.write_text(json.dumps(report, indent=2) + "\n")
    print("PASS", len(reports), flush=True)


if __name__ == "__main__":
    main()
