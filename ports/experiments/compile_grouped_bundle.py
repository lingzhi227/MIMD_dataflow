"""Compile generated grouped CSL in a separate evidence copy; never claims execution."""

import argparse, datetime, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def worker(root):
    os.chdir(root)
    for name, digest in json.loads((root / "provenance.json").read_text())[
        "files"
    ].items():
        assert sha(root / name) == digest
    subprocess.run(json.loads((root / "sdk-command.json").read_text()), check=True)


def main(bundle):
    sys.path.insert(0, str(bundle / "implementation"))
    from integrity import verify_bundle
    from sdk_process import run_sdk

    verify_bundle(bundle)
    s = json.loads((bundle / "schedule.json").read_text())
    assert s["profile"] == "mesh_grouped_gemv.v1"
    root = (
        ROOT
        / "evidence"
        / (
            "grouped-compile-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    root.mkdir()
    for name in [
        "pe.csl",
        "layout.csl",
        "grouped_comm.csl",
        "grouped_routes.csl",
        "schedule.json",
        "semantic.json",
    ]:
        shutil.copyfile(bundle / name, root / name)
    shutil.copyfile(__file__, root / "driver.py")
    p = s["P"]
    params = f"P:{p},Mt:{s['Mt']},Nt:{s['Nt']},pe_num_group:{s['group_size']},root_1st_phase:{s['root_within_group']},root_2nd_phase:{s['global_root']}"
    cmd = [
        "cslc",
        "layout.csl",
        "--arch=wse3",
        f"--fabric-dims={p+7},{p+2}",
        "--fabric-offsets=4,1",
        "--params=" + params,
        "-o=out",
        "--memcpy",
        "--channels=1",
        "--width-west-buf=0",
        "--width-east-buf=0",
    ]
    (root / "sdk-command.json").write_text(json.dumps(cmd, indent=2) + "\n")
    sif = Path(
        "/home/lingzhi/cerebras/sdk/2.10.1/sdk-cbcore-2.10.1-sdk-202606181328-8faf87a26e.sif"
    )
    with sif.open("rb") as f:
        sdksha = hashlib.file_digest(f, "sha256").hexdigest()
    assert sdksha == "fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d"
    (root / "provenance.json").write_text(
        json.dumps(
            dict(
                source_bundle=str(bundle),
                sdk_sha256=sdksha,
                files={p.name: sha(p) for p in root.iterdir() if p.is_file()},
            ),
            indent=2,
        )
        + "\n"
    )
    print(root, flush=True)
    status = dict(compiler_passed=False, execution_run=False)
    try:
        with (root / "sdk.log").open("w") as log:
            run_sdk(
                [
                    "/home/lingzhi/cerebras/sdk/2.10.1/cs_python",
                    str(root / "driver.py"),
                    "--worker",
                    str(root),
                ],
                root,
                dict(os.environ, SINGULARITYENV_CS_TARGET="SDR"),
                log,
                300,
            )
        status["compiler_passed"] = True
    except BaseException as e:
        status["error"] = repr(e)
        raise
    finally:
        (root / "compile-only.json").write_text(json.dumps(status, indent=2) + "\n")
    print("COMPILE ONLY PASS", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle", type=Path)
    p.add_argument("--worker", action="store_true")
    a = p.parse_args()
    worker(a.bundle.resolve()) if a.worker else main(a.bundle.resolve())
