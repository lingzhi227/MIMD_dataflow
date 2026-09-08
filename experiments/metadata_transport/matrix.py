"""Run isolated transport probes in fresh evidence directories; retain failures."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


from pathlib import Path
import datetime, hashlib, json, os, shutil, sys

ROOT = repository_root(__file__)
sys.path.insert(0, str(ROOT / "lib"))
from sdk_process import run_sdk

base = (
    ROOT
    / "validation/evidence"
    / (
        "metadata-matrix-"
        + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
)
base.mkdir()
print(base, flush=True)
cases = [
    (n, o, d, e)
    for e in (2, 3)
    for n in (7, 8, 9, 25, 26, 27)
    for o, d in ((1, 2000), (3, 0))
]
for n, o, d, e in cases:
    out = base / f"n{n}-o{o}-d{d}-e{e}"
    out.mkdir()
    for name in ("layout.csl", "pe.csl", "run.py"):
        shutil.copy2(Path(__file__).parent / name, out / name)
    shutil.copy2(
        ROOT / "runtime/csl/u16_transport.csl", out / "u16_transport.csl"
    )
    (out / "manifest.json").write_text(
        json.dumps(
            dict(
                files={
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in out.iterdir()
                },
                parameters=[n, o, d, e],
            ),
            indent=2,
        )
    )
    with (out / "sdk.log").open("w") as f:
        try:
            run_sdk(
                [
                    "/home/lingzhi/cerebras/sdk/2.10.1/cs_python",
                    str(out / "run.py"),
                    str(n),
                    str(o),
                    str(d),
                    str(e),
                ],
                out,
                dict(os.environ, SINGULARITYENV_CS_TARGET="SDR"),
                f,
                120,
            )
            status = "PASS"
        except Exception as exc:
            status = str(exc)
    print(out.name, status, flush=True)
    if not (out / "out/bin").exists():
        print((out / "sdk.log").read_text()[:1800], flush=True)
        break
