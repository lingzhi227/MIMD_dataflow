"""Independent rectangular MLP source trajectories and original-input arithmetic."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, json, math, sys
from pathlib import Path
import numpy as np
from probe_runtime import read, sha, verify

ROOT = repository_root(__file__)
sys.path.insert(0, str(ROOT / "lib"))
from mesh_twohop import block_index
from mesh_common import pack_tiles, unpack_tiles
from sdk_math_reference import silu_f16


def projection(a, b, p, blocked=False):
    m, k = a.shape
    n = b.shape[1]
    mt, kt, nt = m // p, k // p, n // p
    result = np.zeros((p, p, mt * nt))
    history = np.zeros((p, p, p, mt * nt))
    wide = np.zeros((p, p, mt * nt), np.float32)
    for y in range(p):
        for x in range(p):
            acc = np.zeros((mt, nt))
            total = np.zeros((mt, nt), np.float32)
            for step in range(p):
                if blocked:
                    acc.fill(0)
                block = block_index(p, y, x, step)
                aa = a[y * mt : (y + 1) * mt, block * kt : (block + 1) * kt]
                bb = b[block * kt : (block + 1) * kt, x * nt : (x + 1) * nt]
                for j in range(kt):
                    acc = np.asarray(
                        acc + aa[:, j, None] * bb[None, j, :], np.float16
                    ).astype(float)
                if blocked:
                    total = np.asarray(total + acc.astype(np.float32), np.float32)
                    acc = total.astype(np.float16).astype(float)
                history[y, x, step] = acc.ravel(order="F")
            result[y, x] = acc.ravel(order="F")
            wide[y, x] = total.ravel(order="F")
    return (
        unpack_tiles(result, mt, nt, "F"),
        history.reshape(p, p, -1),
        wide.view(np.uint32),
    )


def standard_product(a, b):
    return np.asarray(
        [
            [math.fsum(float(x) * float(y) for x, y in zip(row, col)) for col in b.T]
            for row in a
        ]
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("probe", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument(
        "--snapshot",
        type=Path,
        help="Read a preserved partial result; never qualifies the source run",
    )
    a = p.parse_args()
    assert not a.output.exists()
    verify(a.probe)
    if a.snapshot is None:
        execution = read(a.probe / "execution.json")
        assert execution["success"] and execution["results_sha256"] == sha(
            a.probe / "results.json"
        )
    provenance = read(a.probe / "provenance.json")
    g = read(a.probe / "geometry.json")
    M, N, F, P = [g[k] for k in ("M", "N", "F", "P")]
    sample = provenance["instrumentation"] == "sampled"
    blocked = provenance.get("blocked_down_accumulation", False)
    result_path = a.snapshot or (a.probe / "results.json")
    r = read(result_path)
    bs = read(a.probe / "logical-inputs.json")
    assert len(bs) == (8 if blocked else 3)
    assert r["runtime_instances"] == 1 and 1 <= len(r["cases"]) <= len(bs)
    complete = bool(r["success"] and len(r["cases"]) == len(bs))
    if a.snapshot is None:
        assert complete
    bits = lambda a: np.asarray(a, np.float16).view(np.uint16)
    rows = []
    for epoch, (b, d) in enumerate(zip(bs, r["cases"])):
        x = np.asarray(b["x"]).reshape(M, N)
        u = np.asarray(b["up_weight"]).reshape(N, F)
        gate_weight = np.asarray(b["gate_weight"]).reshape(N, F)
        down = np.asarray(b["down_weight"]).reshape(F, N)
        up, uh, _ = projection(x, u, P)
        gate, gh, _ = projection(x, gate_weight, P)
        act = np.asarray([silu_f16(v) for v in gate.ravel()]).reshape(M, F)
        hidden = np.asarray(up * act, np.float16).astype(float)
        target, dh, wide = projection(hidden, down, P, blocked=blocked)
        expected = dict(
            up=pack_tiles(up, P, P, "F"),
            activated_gate=pack_tiles(act, P, P, "F"),
            output=pack_tiles(target, P, P, "F"),
        )
        if sample:
            expected.update(
                gate=pack_tiles(gate, P, P, "F"),
                hidden=pack_tiles(hidden, P, P, "F"),
                up_history=uh,
                gate_history=gh,
                down_history=dh,
            )
            mt, nt = M // P, N // P
            expected["gate_left_owner"] = np.asarray(
                [
                    [
                        x[
                            y * mt : (y + 1) * mt,
                            block_index(P, y, z) * nt : (block_index(P, y, z) + 1) * nt,
                        ].ravel(order="F")
                        for z in range(P)
                    ]
                    for y in range(P)
                ]
            )
        else:
            for key in (
                "gate",
                "hidden",
                "up_history",
                "gate_history",
                "down_history",
                "gate_left_owner",
            ):
                np.testing.assert_array_equal(d[key], np.zeros((P, P, 1), np.uint16))
        mismatches = {
            key: int(np.count_nonzero(np.asarray(d[key]) != bits(v)))
            for key, v in expected.items()
        }
        wide_mismatches = None
        if blocked:
            observed = np.asarray(d["wide_accumulator"])
            assert observed.shape == wide.shape and np.all(
                (observed >= 0) & (observed < 2**32)
            )
            wide_mismatches = int(np.count_nonzero(observed != wide))
        su = standard_product(x, u)
        sg = standard_product(x, gate_weight)
        sa = np.asarray([v / (1 + math.exp(-v)) for v in sg.ravel()]).reshape(M, F)
        nominal = standard_product(su * sa, down)
        actual = unpack_tiles(
            np.asarray(d["output"], np.uint16).view(np.float16), M // P, N // P, "F"
        ).astype(float)
        err = actual - nominal
        l2 = float(np.linalg.norm(err)) / max(float(np.linalg.norm(nominal)), 1e-30)
        peak = float(np.max(np.abs(err))) / max(float(np.max(np.abs(nominal))), 1e-30)
        np.testing.assert_array_equal(d["progress"], epoch + 1)
        rows.append(
            dict(
                epoch=epoch,
                target_half_mismatches=mismatches,
                target_f32_accumulator_mismatches=wide_mismatches,
                internal_tensors_observed=sample,
                gate_abs_max=float(np.max(np.abs(gate))),
                standard_relative_l2=l2,
                standard_peak_scaled_error=peak,
                mathematical_accuracy_passed=bool(
                    np.all(np.isfinite(actual)) and l2 <= 0.02 and peak <= 0.03
                ),
            )
        )
    passed = all(
        not any(v["target_half_mismatches"].values())
        and v["target_f32_accumulator_mismatches"] in (None, 0)
        and v["mathematical_accuracy_passed"]
        for v in rows
    )
    report = dict(
        passed=passed and complete and a.snapshot is None,
        completed_calls_valid=passed,
        complete=complete,
        snapshot_only=a.snapshot is not None,
        blocked_down_accumulation=blocked,
        shared_precision_library=blocked,
        preserve_completed_left_ownership=provenance[
            "preserve_completed_left_ownership"
        ],
        cases=rows,
        new_sdk_execution=False,
        scope="Supplied bounded activations and independent rectangular weights. Target half staged contractions/source SiLU checked separately from original-input math.fsum/exp arithmetic. No RMS/residual/full-model or HLS qualification. Original unmodified gate-branch pointer reset may fail; optional repair is explicit.",
        hashes={
            str(p): sha(p)
            for p in [
                a.probe / "provenance.json",
                result_path,
                Path(__file__),
                ROOT / "lib/Conversion/mesh_twohop.py",
                ROOT / "lib/Conversion/mesh_common.py",
                ROOT / "lib/Numerics/sdk_math_reference.py",
                ROOT / "lib/Numerics/binary16.py",
            ]
        },
    )
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if provenance["preserve_completed_left_ownership"]:
        assert passed
