"""Typed compute/storage boundaries, independent of spatial schedule selection.

Acceptance here does not admit a backend. Mixed policies must also prove their
layout, transport width, numerical range, resource lifetime and execution gates.
"""

import math
from frontend import check


def verify(module):
    by = {n["id"]: n for n in module["nodes"]}
    check(len(by) == len(module["nodes"]), "unique precision graph values")
    result = []
    for node in module["nodes"]:
        policy = node.get("precision")
        if policy is None:
            check(
                node.get("accumulation") != "f32",
                "f32 accumulation requires explicit compute/storage types",
            )
            continue
        op = node["op"]
        dtype = node.get("dtype", "f32")
        check(
            op in ("add", "matmul", "softmax", "rmsnorm"),
            "explicit precision operation",
        )
        check(
            policy == dict(compute="f32", storage=dtype, explicit=True)
            and dtype in ("f16", "f32"),
            "explicit f32 compute and matching result storage",
        )
        check(all(v in by for v in node["inputs"]), "defined precision operands")
        operands = [by[v] for v in node["inputs"]]
        check(
            all(v.get("dtype", "f32") in ("f16", "f32") for v in operands),
            "floating precision operands",
        )
        flow = node.get("dataflow", {})
        if op == "matmul":
            check(len(operands) == 2, "mixed contraction operands")
            a, b = operands
            check(
                a["shape"][1] == b["shape"][0]
                and node["shape"] == [a["shape"][0], b["shape"][1]],
                "mixed contraction shapes",
            )
            check(
                node.get("accumulation") == flow.get("accumulation") == "f32",
                "typed contraction and pragma accumulation agreement",
            )
            check(
                "block_size" not in node,
                "full f32 accumulation is not a half-block merge",
            )
        elif op == "add":
            check(
                len(operands) == 2
                and all(v["shape"] == node["shape"] for v in operands),
                "mixed residual shapes",
            )
            check(
                "accumulation" not in flow,
                "elementwise addition has no reduction policy",
            )
        else:
            check(
                flow.get("accumulation") == "f32" and flow.get("math") == "sdk_float",
                "typed normalization and pragma arithmetic agreement",
            )
            check(
                len(operands) == (1 if op == "softmax" else 2),
                "normalization precision operands",
            )
            check(
                node["shape"] == operands[0]["shape"], "normalization preserves shape"
            )
            value = node["scale" if op == "softmax" else "epsilon"]
            check(
                type(value) in (int, float) and math.isfinite(value) and value > 0,
                "finite positive normalization parameter",
            )
            if op == "softmax":
                check(dtype == "f32", "wide softmax retains probabilities in f32")
            else:
                check(
                    operands[1]["shape"] == [1, node["shape"][1]],
                    "RMS feature weight shape",
                )
        result.append(
            dict(
                node=node["id"],
                operator=op,
                compute_dtype="f32",
                input_storage=[v.get("dtype", "f32") for v in operands],
                output_storage=dtype,
                narrowing=dtype == "f16",
            )
        )
    return result
