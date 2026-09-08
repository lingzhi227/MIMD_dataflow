"""Common contracts and host tile packing for matrix dataflow backends."""

import copy
from frontend import check


def verify_matmul(module, epochs, bound, *, compute_modes=("vector", "scalar")):
    m = copy.deepcopy(module)
    check(
        [n["op"] for n in m["nodes"]] == ["input", "input", "matmul", "output"],
        "mesh requires two inputs, matmul, output",
    )
    a, b, op, out = m["nodes"]
    check(
        len({n["id"] for n in m["nodes"]}) == 4 and a["host"] != b["host"],
        "mesh unique ids/ports",
    )
    check(
        op["inputs"] == [a["id"], b["id"]] and out["inputs"] == [op["id"]],
        "mesh dependencies",
    )
    check(
        not m["states"] and not any("place" in n for n in m["nodes"]),
        "mesh owns placement",
    )
    check(
        all(
            len(n["shape"]) == 2
            and all(type(v) is int and 0 < v <= 512 for v in n["shape"])
            for n in (a, b, op)
        ),
        "mesh matrix dimensions 1..512",
    )
    check(
        a["shape"][1] == b["shape"][0]
        and op["shape"] == [a["shape"][0], b["shape"][1]],
        "mesh matmul shapes",
    )
    check(
        type(epochs) is int
        and 1 <= epochs <= 16
        and type(bound) is int
        and 0 <= bound <= 32767,
        "mesh execution bounds",
    )
    d = op["dataflow"]
    check(
        d["fp"] == "relaxed" and d["compute"] in compute_modes,
        "mesh explicit floating policy",
    )
    out["shape"] = op["shape"][:]
    for n in m["nodes"]:
        n["interval"] = None
    m.update(epochs=epochs, input_bound=bound)
    return m


def pack_tiles(matrix, rows, cols, order="C"):
    import numpy as np

    matrix = np.asarray(matrix, np.float32)
    h, w = matrix.shape
    check(h % rows == 0 and w % cols == 0, "tile packing divisibility")
    tiles = matrix.reshape(rows, h // rows, cols, w // cols).transpose(0, 2, 1, 3)
    check(order in ("C", "F"), "tile memory order")
    if order == "F":
        tiles = tiles.transpose(0, 1, 3, 2)
    return tiles.reshape(rows, cols, -1)


def unpack_tiles(tiles, mt, nt, order="C"):
    import numpy as np

    tiles = np.asarray(tiles)
    rows, cols, words = tiles.shape
    check(words == mt * nt and order in ("C", "F"), "tile unpacking extent/order")
    blocks = (
        tiles.reshape(rows, cols, mt, nt)
        if order == "C"
        else tiles.reshape(rows, cols, nt, mt).transpose(0, 1, 3, 2)
    )
    return blocks.transpose(0, 2, 1, 3).reshape(rows * mt, cols * nt)


def validate_sdk_options(options):
    check(
        set(options) == {"suppress_trace", "num_threads", "dump_core"},
        "SDK runtime option keys",
    )
    check(
        type(options["suppress_trace"]) is bool and type(options["dump_core"]) is bool,
        "SDK runtime boolean options",
    )
    check(
        type(options["num_threads"]) is int and 1 <= options["num_threads"] <= 64,
        "SDK simulator thread bound",
    )


def sdk_runtime(root, artifact="out"):
    """Use explicit simulator controls only when frozen into the build bundle."""
    import json
    from pathlib import Path
    from cerebras.sdk.runtime.sdkruntimepybind import (
        SdkRuntime,
        SimfabConfig,
        SdkTarget,
        get_platform,
    )

    path = Path(root) / "runtime-options.json"
    if not path.exists():
        return SdkRuntime(artifact)
    options = json.loads(path.read_text())
    validate_sdk_options(options)
    platform = get_platform(None, SimfabConfig(**options), SdkTarget.WSE3)
    return SdkRuntime(artifact, platform)
