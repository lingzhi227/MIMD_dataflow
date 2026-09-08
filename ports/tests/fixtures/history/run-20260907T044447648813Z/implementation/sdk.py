"""SDK 2.10.1/WSE3 binding; host transports inputs/final outputs only."""

import argparse, json, os
from pathlib import Path

os.environ["CS_TARGET"] = "SDR"
from integrity import verify_bundle
import numpy as np
from cerebras.sdk.runtime.sdkruntimepybind import (
    SdkLayout,
    SdkTarget,
    SdkRuntime,
    SimfabConfig,
    get_platform,
    Edge,
    Route,
    RoutingPosition,
)


def route(incoming):
    r = RoutingPosition()
    if incoming:
        r.set_output([Route.RAMP])
    else:
        r.set_input([Route.RAMP])
    return [r]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("directory")
    args = p.parse_args()
    root = Path(args.directory).resolve()
    verify_bundle(root)
    s = json.loads((root / "schedule.json").read_text())
    if s.get("profile") == "mesh_mlp.v1":
        from mesh_mlp_sdk import run as mlp_run

        return mlp_run(root)
    if s.get("profile") == "mesh_attention.v1":
        from mesh_attention_sdk import run as attention_run

        return attention_run(root)
    if s.get("profile") == "mesh_score_softmax.v1":
        from mesh_score_softmax_sdk import run as resident_run

        return resident_run(root)
    if s.get("profile") == "mesh_device_matmul.v1":
        from mesh_device_matmul_sdk import run as device_run

        return device_run(root)
    if s.get("profile") == "mesh_score.v1":
        from mesh_score_sdk import run as score_run

        return score_run(root)
    if s.get("profile") == "mesh_pair_rotation.v1":
        from mesh_pair_rotation_sdk import run as pair_run

        return pair_run(root)
    if s.get("profile") == "mesh_swiglu.v1":
        from mesh_swiglu_sdk import run as gating_run

        return gating_run(root)
    if s.get("profile") == "mesh_normalized_fanout.v1":
        from mesh_normalized_fanout_sdk import run as fanout_run

        return fanout_run(root)
    if s.get("profile") == "mesh_normalized_matmul.v1":
        from mesh_normalized_matmul_sdk import run as resident_run

        return resident_run(root)
    if s.get("profile") == "mesh_softmax.v1":
        from mesh_softmax_sdk import run as softmax_run

        return softmax_run(root)
    if s.get("profile") == "mesh_rms.v1":
        from mesh_rms_sdk import run as rms_run

        return rms_run(root)
    if s.get("profile") == "mesh_fft.v1":
        from mesh_fft_sdk import run as fft_run

        return fft_run(root)
    if s.get("profile") == "mesh_grouped_gemv.v1":
        from grouped_gemv_sdk import run as grouped_run

        return grouped_run(root)
    if s.get("profile") == "mesh_twohop.v1":
        from twohop_sdk import run

        return run(root)
    if s.get("profile") == "mesh_reduction.v1":
        from mesh_reduction_sdk import run as reduction_run

        return reduction_run(root)
    if s.get("profile") in ("mesh_cg.v1", "mesh_power.v1"):
        from mesh_cg_sdk import run

        return run(root)
    if s.get("profile") == "mesh_spmv.v1":
        from mesh_spmv_sdk import run

        return run(root)
    if s.get("profile") == "mesh_qr.v1":
        from mesh_qr_sdk import run

        return run(root)
    if s.get("profile") == "mesh_lu.v1":
        from mesh_lu_sdk import run

        return run(root)
    if s.get("profile") == "mesh_cholesky.v1":
        from mesh_cholesky_sdk import run

        return run(root)
    if s.get("profile") in ("mesh_gemm.v1", "mesh_cannon.v1"):
        from mesh_gemm_sdk import run

        return run(root)
    if s.get("profile") == "mesh_gemv.v1":
        from mesh_gemv_sdk import run

        return run(root)
    if s.get("profile") == "grid.v1":
        from grid_sdk import run

        return run(root)
    batches = json.loads((root / "batches.json").read_text())
    loads = [n for n in s["nodes"] if n["op"] == "input"]
    stores = [n for n in s["nodes"] if n["op"] == "output"]
    if len(batches) != s["epochs"]:
        raise ValueError("epoch count")
    for b in batches:
        if set(b) != {n["host"] for n in loads}:
            raise ValueError("input ports")
        for n in loads:
            v = b[n["host"]]
            if len(v) != n["output_size"] or any(
                type(x) not in (int, float)
                or not np.isfinite(x)
                or abs(x) > s["input_bound"]
                for x in v
            ):
                raise ValueError("input shape/bounds")
    os.chdir(root)
    platform = get_platform(None, SimfabConfig(dump_core=True), SdkTarget.WSE3)
    layout = SdkLayout(platform)
    ports = {}
    regions = {}
    for n in s["nodes"]:
        name = n["id"]
        code = layout.create_code_region(str(root / (name + ".csl")), name, 1, 1)
        code.place(*n["place"])
        regions[name] = code
        for i, size in enumerate(n["wire_input_sizes"]):
            key = "rx" + str(i)
            color = code.color(key)
            code.set_param_all(color)
            edge = Edge.TOP if n["op"] == "input" or i == 1 else Edge.LEFT
            ports[name, key] = code.create_input_port(
                color, edge, route(True), size, key
            )
        for i in range(n["send_ports"]):
            key = "tx" + str(i)
            color = code.color(key)
            code.set_param_all(color)
            edge = (
                Edge.TOP
                if n["op"] == "output"
                else Edge.RIGHT if i == 0 else Edge.BOTTOM
            )
            ports[name, key] = code.create_output_port(
                color, edge, route(False), n["wire_output_size"], key
            )
    for n in s["nodes"]:
        for i, c in enumerate(n["consumers"]):
            layout.connect(
                ports[n["id"], "tx" + str(i)], ports[c["node"], "rx" + str(c["input"])]
            )
    inputs = {
        n["host"]: layout.create_input_stream(ports[n["id"], "rx0"]) for n in loads
    }
    outputs = {
        n["host"]: layout.create_output_stream(ports[n["id"], "tx0"]) for n in stores
    }
    print("SDK COMPILE START", flush=True)
    artifact = layout.compile(
        out_prefix="out",
        cslc_prefix="/cb/toolchains/cslang/rel-sdk-2.10.0/202604012315-1813-1394a6ea",
    )
    print("SDK COMPILE PASS", flush=True)
    runner = SdkRuntime(artifact, platform, memcpy_required=False)
    report = {
        "success": False,
        "cases": [],
        "diagnostics": {},
        "artifact": str(artifact),
    }

    def save():
        (root / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    runner.load()
    runner.run()
    try:
        for e, batch in enumerate(batches):
            retained = []
            for name, stream in inputs.items():
                values = batch[name]
                data = np.array(
                    values + [0.0] * (len(values) % 2), dtype=np.float32
                ).view(np.uint32)
                retained.append(data)
                runner.send(stream, data, nonblock=True)
            values = {}
            for n in stores:
                out = np.zeros(n["wire_output_size"], dtype=np.uint32)
                runner.receive(outputs[n["host"]], out, len(out), nonblock=False)
                if any(out[n["output_size"] :]):
                    raise ValueError("nonzero transport padding")
                values[n["host"]] = out.view(np.float32)[: n["output_size"]].tolist()
            report["cases"].append(values)
            save()
            print("EPOCH", e + 1, "RECEIVED", flush=True)
    finally:
        runner.stop()
    for n in s["nodes"]:
        x, y = n["place"]
        fields = [
            "history",
            "received_a",
            "produced",
            "completed",
            "epochs",
            "inflight",
        ]
        if len(n["input_sizes"]) == 2:
            fields.append("received_b")
        if n["op"] == "accumulate":
            fields.append("state")
        report["diagnostics"][n["id"]] = {
            f: runner.read_symbol(
                x, y, f, dtype="uint16" if f == "epochs" else "uint32"
            ).tolist()
            for f in fields
        }
    report["success"] = True
    save()
    print("SDK EXECUTION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
