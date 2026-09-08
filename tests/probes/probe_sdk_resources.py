"""Ask the installed SDK compiler about concrete WSE3 resource boundaries."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, datetime, json, os, subprocess, sys
from pathlib import Path

ROOT = repository_root(__file__)
SDK = "/home/lingzhi/cerebras/sdk/2.10.1/cs_python"
p = argparse.ArgumentParser()
p.add_argument("--worker")
a = p.parse_args()
if a.worker:
    from cerebras.sdk.runtime.sdkruntimepybind import (
        SdkLayout,
        SdkTarget,
        SimfabConfig,
        get_platform,
    )

    root = Path(a.worker).resolve()
    os.chdir(root)
    layout = SdkLayout(get_platform(None, SimfabConfig(), SdkTarget.WSE3))
    region = layout.create_code_region(str(root / "probe.csl"), "probe", 1, 1)
    region.place(4, 4)
    layout.compile(
        out_prefix="out",
        cslc_prefix="/cb/toolchains/cslang/rel-sdk-2.10.0/202604012315-1813-1394a6ea",
    )
    raise SystemExit(0)
root = (
    ROOT
    / "validation/evidence"
    / (
        "resource-probes-"
        + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
)
root.mkdir()
probes = [
    ("valid_edges", 23, 7, 7, 30, True),
    ("color_24", 24, 7, 7, 30, False),
    ("input_queue_8", 23, 8, 7, 30, False),
    ("output_queue_8", 23, 7, 8, 30, False),
    ("local_task_31", 23, 7, 7, 31, False),
    ("local_task_32", 23, 7, 7, 32, False),
    ("local_task_7", 23, 2, 2, 7, False),
    ("microthread_7", 23, 7, 7, 30, True, 7),
    ("microthread_8", 23, 7, 7, 30, False, 8),
]
report = {
    "sdk": "2.10.1",
    "target": "WSE3",
    "kind": "compile probes only; not runtime liveness/performance evidence",
    "probes": [],
}
for name, color, iq, oq, tid, expected, *extra in probes:
    ut = extra[0] if extra else 0
    dest = root / name
    dest.mkdir()
    (dest / "probe.csl").write_text(f"""task receive(data:u32) void {{}}
var buffer=@zeros([1]f32);
const source=@get_dsd(mem1d_dsd,.{{.tensor_access=|i|{{1}}->buffer[i]}});
const target=@get_dsd(fabout_dsd,.{{.extent=1,.output_queue=@get_output_queue({oq})}});
task tick() void {{@fmovs(target,source,.{{.async=true,.ut_id=@get_ut_id({ut})}});}}
comptime{{
 @comptime_assert(@is_arch("wse3"));
 const q=@get_input_queue({iq});
 @initialize_queue(q,.{{.color=@get_color({color})}});
 @initialize_queue(@get_output_queue({oq}),.{{.color=@get_color({color})}});
 @bind_data_task(receive,@get_data_task_id(q));
 @bind_local_task(tick,@get_local_task_id({tid}));
}}
""")
    env = dict(os.environ, SINGULARITYENV_CS_TARGET="SDR")
    result = subprocess.run(
        [SDK, str(Path(__file__).resolve()), "--worker", str(dest)],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
    )
    (dest / "compiler.log").write_text(result.stdout)
    row = {
        "name": name,
        "compiled": result.returncode == 0,
        "expected": expected,
        "compiler_error_diagnostic": "error:" in result.stdout.lower(),
        "log": str((dest / "compiler.log").relative_to(ROOT)),
    }
    report["probes"].append(row)
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(row, flush=True)
report["expected_checks_passed"] = all(
    r["expected"] is None
    or (
        r["compiled"] == r["expected"]
        and (r["compiled"] or r["compiler_error_diagnostic"])
    )
    for r in report["probes"]
)
(root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(root)

if not report["expected_checks_passed"]:
    raise SystemExit(1)
