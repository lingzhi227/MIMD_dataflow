"""SDK dyadic statistic scaling / RMS mean mode, real overflow and warm handoffs."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, datetime, difflib, json, shutil, sys, warnings
from pathlib import Path
import numpy as np

ROOT = repository_root(__file__)
sys.path[:0] = (
    [str(Path(__file__).resolve().parent)]
    if (Path(__file__).resolve().parent / "probe_runtime.py").exists()
    else [str(ROOT / "experiments"), str(ROOT / "lib")]
)
from probe_runtime import execute, mesh_half_worker, read, sha, verify

PORTS = dict(
    timing=18,
    Z=96,
    scratch=96,
    local_stats=6,
    sum_box=6,
    mean_box=6,
    after_box=6,
    normalized=96,
    legacy_normalized=96,
    mean_send=4,
    mean_reduced=4,
    progress=1,
    callbacks=1,
    legacy_valid=1,
    queues=2,
    skew=1,
)
DIVISORS = (256, 256, 1, 256, 128, 256, 512, 256)


def prepare(participants=8):
    assert participants in (8, 16)
    nt = 256 // participants
    ports = dict(PORTS)
    for name in ("Z", "scratch", "normalized", "legacy_normalized"):
        ports[name] = 3 * nt
    from sdk_math_reference import rms_inverse_f16

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root = ROOT / "validation/evidence" / ("sdk-axis-mean-" + stamp)
    root.mkdir()
    base = ROOT / "validation/evidence/sdk-axis-max-20260907T213834656626Z"
    verify(base)
    assert read(base / "execution.json")["success"]
    for n in ("runtime-options.json", "sdk-command.json"):
        shutil.copyfile(base / n, root / n)
    for n in ("sdk_axis_reduce.csl", "sdk_axis_mean.csl", "batched_rms_local.csl"):
        shutil.copyfile(ROOT / "runtime/csl" / n, root / n)
    layout = (base / "layout.csl").read_text()
    command = read(root / "sdk-command.json")
    command = [
        (
            f"--fabric-dims={participants+7},{participants+2}"
            if v.startswith("--fabric-dims=")
            else v
        )
        for v in command
    ]
    (root / "sdk-command.json").write_text(json.dumps(command) + "\n")
    layout = (
        layout.replace(
            ".width=8,.height=8", f".width={participants},.height={participants}"
        )
        .replace(
            "@set_rectangle(8,8)", f"@set_rectangle({participants},{participants})"
        )
        .replace("@range(u16,8)", f"@range(u16,{participants})")
        .replace(".memcpy_params=", f".Nt={nt},.memcpy_params=")
    )
    layout = layout[: layout.index('    @export_name("A"')]
    pe = (Path(__file__).parent / "pe.csl").read_text()
    for name in ports:
        dtype = (
            "f32"
            if name in ("mean_send", "mean_reduced")
            else (
                "u16"
                if name
                in ("progress", "callbacks", "legacy_valid", "queues", "skew", "timing")
                else "f16"
            )
        )
        variable = {"mean_send": "mean_op.send", "mean_reduced": "mean_op.reduced"}.get(
            name, name
        )
        pe += f"\nvar export_{name}:[*]{dtype}=&{variable};\n"
        layout += f'    @export_name("{name}",[*]{dtype},true);\n'
    pe += (
        "\ncomptime {\n"
        + "".join(f' @export_symbol(export_{n},"{n}");\n' for n in ports)
        + ' @export_symbol(init_task,"init_task"); @export_symbol(hls_main,"hls_main");\n}\n'
    )
    layout += (
        ' @export_name("init_task",fn()void); @export_name("hls_main",fn()void);\n}\n'
    )
    (root / "pe.csl").write_text(pe)
    (root / "layout.csl").write_text(layout)
    schema = dict(
        rows=participants,
        cols=participants,
        inputs=dict(Z=3 * nt),
        outputs=ports,
        output_word_bits=dict(mean_send=32, mean_reduced=32),
        immutable=["Z"],
        progress="progress",
        initialize="init_task",
        launch="hls_main",
    )
    q = lambda a: np.asarray(a, np.float16).astype(float)

    def sum32(a, axis):
        v = np.moveaxis(np.asarray(a, np.float32), axis, 0)
        total = v[-1].copy()
        for row in v[-2::-1]:
            total = np.asarray(total + row, np.float32)
        return np.repeat(np.expand_dims(total, axis), participants, axis=axis)

    inputs = []
    expected = []
    math_checks = []
    skew = np.zeros((participants, participants, 1), np.uint16)
    for e, divisor in enumerate(DIVISORS):
        z = np.empty((participants, participants, 3, nt))
        axis = 1 if e % 2 == 0 else 0
        count = 3 if e % 2 == 0 else 4
        for y in range(participants):
            for x in range(participants):
                for b in range(3):
                    for j in range(nt):
                        if e == 0:
                            value = 33.0
                        elif e == 1:
                            value = 34.5625
                        elif e == 2:
                            value = 0.0
                        elif e == 5:
                            value = (
                                (1 if (j + x + y) % 2 else -1)
                                * (1 + b + (j % 3 == 0))
                                / 4096
                            )
                        elif e == 6:
                            value = 2.0 + (b * 8 + (x + 3 * y + j) % 8) / 64
                        else:
                            value = (((x * 17 + y * 5 + b * 3 + j + e) % 33) - 16) / 32
                        z[y, x, b, j] = value
        scratch = q(z * z)
        local = np.zeros((participants, participants, 4))
        for j in range(nt):
            local[:, :, :3] = q(local[:, :, :3] + scratch[:, :, :, j])
        assert np.all(np.isfinite(local))
        totals = sum32(local, axis)
        with np.errstate(over="ignore"):
            narrowed = q(totals)
        send = np.asarray(local / float(divisor), np.float32)
        mean32 = sum32(send, axis)
        mean = q(mean32)
        assert np.all(np.isfinite(mean))
        box = lambda fill: np.full((participants, participants, 6), fill, dtype=float)
        stats = box(0)
        stats[:, :, 0] = 17
        stats[:, :, 5] = -19
        stats[:, :, 1:5] = local
        sb = box(-13)
        sb[:, :, 1 : count + 1] = narrowed[:, :, :count]
        mb = box(-13)
        mb[:, :, 1 : count + 1] = mean[:, :, :count]
        after = box(-23)
        other_count = 4 if count == 3 else 3
        with np.errstate(over="ignore"):
            after[:, :, 1 : other_count + 1] = q(sum32(local, 1 - axis))[
                :, :, :other_count
            ]
        norm = np.empty_like(z)
        legacy = np.zeros_like(z)
        for y in range(participants):
            for x in range(participants):
                for b in range(3):
                    norm[y, x, b] = q(
                        z[y, x, b] * rms_inverse_f16(mean[y, x, b], 1, 1e-6)
                    )
                    if e >= 2:
                        legacy[y, x, b] = q(
                            z[y, x, b] * rms_inverse_f16(narrowed[y, x, b], 256, 1e-6)
                        )
                for phase in range(3):
                    delay = 1 + (x + 3 * y + e + phase) % 17
                    skew[y, x, 0] += delay * (delay + 1) // 2
        true_sum = np.sum(z * z, axis=(axis, 3), keepdims=True)
        ideal = z / np.sqrt(true_sum / divisor + 1e-6)
        err = norm - ideal
        relative_l2 = float(np.linalg.norm(err) / max(np.linalg.norm(ideal), 1e-30))
        relative_peak = float(np.max(np.abs(err)) / max(np.max(np.abs(ideal)), 1e-30))
        math_checks.append(
            dict(
                epoch=e,
                divisor=divisor,
                relative_l2=relative_l2,
                relative_peak=relative_peak,
                standard_rms=divisor == 256,
            )
        )
        inputs.append(dict(Z=z.reshape(participants, participants, 3 * nt).tolist()))
        data = dict(
            Z=z.reshape(participants, participants, 3 * nt),
            scratch=scratch.reshape(participants, participants, 3 * nt),
            local_stats=stats,
            sum_box=sb,
            mean_box=mb,
            after_box=after,
            normalized=norm.reshape(participants, participants, 3 * nt),
            legacy_normalized=legacy.reshape(participants, participants, 3 * nt),
        )
        words = {
            k: np.asarray(v, np.float16).view(np.uint16).tolist()
            for k, v in data.items()
        }
        words.update(
            mean_send=send.view(np.uint32).tolist(),
            mean_reduced=mean32.view(np.uint32).tolist(),
            progress=np.full((participants, participants, 1), e + 1).tolist(),
            callbacks=np.full((participants, participants, 1), 3 * (e + 1)).tolist(),
            legacy_valid=np.full((participants, participants, 1), int(e >= 2)).tolist(),
            skew=skew.copy().tolist(),
        )
        expected.append(words)
    for name, value in [
        ("inputs.json", inputs),
        ("expected.json", expected),
        ("schema.json", schema),
        ("predicted-math.json", math_checks),
    ]:
        (root / name).write_text(json.dumps(value, allow_nan=False) + "\n")
    old = (
        ROOT
        / "benchmarks/inference/waferllm/projected_cache_attention_3x256x512_8x8/run-20260907T235739454777Z/batched_rms_local.csl"
    )
    (root / "rms-mean-api.patch").write_text(
        "".join(
            difflib.unified_diff(
                old.read_text().splitlines(True),
                (root / "batched_rms_local.csl").read_text().splitlines(True),
                fromfile="qualified-sum-default",
                tofile="additive-mean-statistic",
            )
        )
    )
    for src, name in [
        (Path(__file__), "driver.py"),
        (ROOT / "experiments/probe_runtime.py", "probe_runtime.py"),
        (ROOT / "lib/Runtime/sdk_process.py", "sdk_process.py"),
        (ROOT / "lib/Numerics/sdk_math_reference.py", "sdk_math_reference.py"),
    ]:
        shutil.copyfile(src, root / name)
    (root / "provenance.json").write_text(
        json.dumps(
            dict(
                scope="Primitive only: actual half local RMS squares/sums, ordinary SUM overflow witnesses, mean before narrow on shared SDK planes, in-place offset/canaries, both axes,3/4extents, changed divisors, default RMS finite cases, skew and8warm calls. No complete HLS/FFN qualification.",
                base_probe=str(base.relative_to(ROOT)),
                base_provenance_sha256=sha(base / "provenance.json"),
                qualified_sum_module_sha256=sha(old),
                files={p.name: sha(p) for p in root.iterdir() if p.is_file()},
            ),
            indent=2,
        )
        + "\n"
    )
    print(root.relative_to(ROOT), flush=True)
    print(math_checks, flush=True)


def run(root):
    root = root.resolve()
    execute(root, 1800)
    r, e = read(root / "results.json"), read(root / "expected.json")
    assert r["success"] and len(r["cases"]) == len(e) == 8
    for epoch, (actual, want) in enumerate(zip(r["cases"], e)):
        for name, value in want.items():
            np.testing.assert_array_equal(
                actual[name], value, err_msg=f"{epoch} {name}"
            )
        assert np.all((np.asarray(actual["queues"]) & 60) == 60)
    checks = read(root / "predicted-math.json")
    passed = all(
        c["relative_l2"] <= 0.02 and c["relative_peak"] <= 0.03 for c in checks
    )
    (root / "mean-review.json").write_text(
        json.dumps(
            dict(
                passed=passed,
                raw_words_passed=True,
                epochs=8,
                checks=checks,
                limits=dict(relative_l2=0.02, relative_peak=0.03),
                results_sha256=sha(root / "results.json"),
                provenance_sha256=sha(root / "provenance.json"),
                scope="Actual raw trajectory equals frozen predictions; normalized results then satisfy original-input expression x/sqrt(sum(x*x)/divisor+epsilon). This is RMS only when divisor256. Default sum RMS skipped for two intentional overflow witnesses; no full application qualification.",
            ),
            indent=2,
        )
        + "\n"
    )
    assert passed, "fixed mathematical gate failed; preserve run"
    print("SDK MEAN/RMS PRIMITIVE PASS", root, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--participants", type=int, default=8)
    p.add_argument("--execute", type=Path)
    p.add_argument("--worker", type=Path)
    a = p.parse_args()
    if a.prepare:
        prepare(a.participants)
    elif a.worker:
        mesh_half_worker(a.worker.resolve())
    elif a.execute:
        run(a.execute)
