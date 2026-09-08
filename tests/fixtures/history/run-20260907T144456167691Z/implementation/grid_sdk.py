"""SDK binding: one resident region per pencil, neighbor ports, initial/final IO only."""

import json, os
from pathlib import Path
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


def run(root):
    root = Path(root).resolve()
    s = json.loads((root / "schedule.json").read_text())
    m = json.loads((root / "semantic.json").read_text())
    batches = json.loads((root / "batches.json").read_text())
    os.chdir(root)
    platform = get_platform(None, SimfabConfig(dump_core=True), SdkTarget.WSE3)
    layout = SdkLayout(platform)
    ports = {}
    inputs = {}
    outputs = {}
    edges = {
        "west": Edge.LEFT,
        "east": Edge.RIGHT,
        "south": Edge.TOP,
        "north": Edge.BOTTOM,
    }
    opposite = {"west": "east", "east": "west", "south": "north", "north": "south"}
    for n in s["nodes"]:
        key = n["id"]
        region = layout.create_code_region(str(root / (key + ".csl")), key, 1, 1)
        region.place(*n["place"])
        io_extra = (
            [
                ("tx_init", False, n["forward_size"], Edge.BOTTOM),
                ("rx_collect", True, n["collect_size"], Edge.BOTTOM),
            ]
            if n["forward_size"]
            else []
        )
        for tag, incoming, size, edge in (
            io_extra
            + [
                ("rx_host", True, n["ingress_size"], Edge.TOP),
                ("tx_host", False, n["egress_size"], Edge.TOP),
            ]
            + [
                (prefix + d, incoming, s["wire_z"], edges[d])
                for d in n["neighbors"]
                for prefix, incoming in [("rx_", True), ("tx_", False)]
            ]
        ):
            color = region.color(tag)
            region.set_param_all(color)
            method = region.create_input_port if incoming else region.create_output_port
            ports[key, tag] = method(color, edge, route(incoming), size, tag)
        if n["tile"][1] == 0:
            inputs[key] = layout.create_input_stream(ports[key, "rx_host"])
            outputs[key] = layout.create_output_stream(ports[key, "tx_host"])
    for n in s["nodes"]:
        if n["forward_size"]:
            x, y = n["tile"]
            child = f"p{x}_{y+1}"
            layout.connect(ports[n["id"], "tx_init"], ports[child, "rx_host"])
            layout.connect(ports[child, "tx_host"], ports[n["id"], "rx_collect"])
        for d, peer in n["neighbors"].items():
            layout.connect(ports[n["id"], "tx_" + d], ports[peer, "rx_" + opposite[d]])
    print("GRID SDK COMPILE START", len(s["nodes"]), "PEs", flush=True)
    artifact = layout.compile(
        out_prefix="out",
        cslc_prefix="/cb/toolchains/cslang/rel-sdk-2.10.0/202604012315-1813-1394a6ea",
    )
    print("GRID SDK COMPILE PASS", flush=True)
    runner = SdkRuntime(artifact, platform, memcpy_required=False)
    report = {
        "success": False,
        "cases": [],
        "diagnostics": {},
        "artifact": str(artifact),
    }

    def save():
        (root / "results.json").write_text(json.dumps(report) + "\n")

    runner.load()
    runner.run()
    z = s["grid"]["z"]
    ny = s["grid"]["y"]
    try:
        for epoch, batch in enumerate(batches):
            retained = []
            field = batch[m["nodes"][0]["host"]]
            coeff = batch[m["nodes"][1]["host"]]
            for n in s["nodes"]:
                x, y = n["tile"]
                if y:
                    continue
                v = []
                for row in range(ny):
                    at = (x * ny + row) * z
                    packet = field[at : at + z] + coeff
                    v.extend(packet + [0.0] * (len(packet) % 2))
                data = np.asarray(v, dtype=np.float32).view(np.uint32)
                retained.append(data)
                runner.send(inputs[n["id"]], data, nonblock=True)
            final = [0.0] * len(field)
            for n in s["nodes"]:
                x, y = n["tile"]
                if y:
                    continue
                data = np.zeros(n["egress_size"], dtype=np.uint32)
                runner.receive(outputs[n["id"]], data, len(data), nonblock=False)
                for row in range(ny):
                    packet = data[row * s["wire_z"] : (row + 1) * s["wire_z"]]
                    if any(packet[z:]):
                        raise ValueError("padding")
                    at = (x * ny + row) * z
                    final[at : at + z] = packet.view(np.float32)[:z].tolist()
            report["cases"].append({m["nodes"][3]["host"]: final})
            save()
            print("GRID EPOCH", epoch + 1, "PASS transport", flush=True)
    finally:
        runner.stop()
    for n in s["nodes"]:
        x, y = n["place"]
        report["diagnostics"][n["id"]] = {
            f: runner.read_symbol(
                x, y, f, dtype="uint16" if f in ("epochs", "step") else "uint32"
            ).tolist()
            for f in (
                "history",
                "epochs",
                "step",
                "received",
                "sent",
                "host_received",
                "host_sent",
                "forwarded",
                "collected",
            )
        }
    report["success"] = True
    save()
