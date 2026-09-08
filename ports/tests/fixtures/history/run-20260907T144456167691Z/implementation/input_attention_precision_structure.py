"""Canonical graph roles for the mixed resident chain.

The half shadow checks only established topology/layout restrictions. Its
arithmetic ranges are NOT a certificate for the actual mixed graph.
"""

import copy
from frontend import check
from precision_contracts import verify as precision_verify
from mesh_input_attention import verify as half_verify


def shadow(module):
    m = copy.deepcopy(module)
    for node in m["nodes"]:
        if "precision" in node:
            node.pop("precision")
            node["dtype"] = "f16"
            if node.get("accumulation") == "f32":
                node.pop("accumulation")
            flow = node.get("dataflow", {})
            if node["op"] == "matmul":
                flow.pop("accumulation", None)
            elif node["op"] in ("softmax", "rmsnorm"):
                flow.update(accumulation="f16", math="sdk_half")
    return m


def canonical(module, epochs, bound):
    facts = precision_verify(module)
    structure = half_verify(shadow(module), epochs, bound)
    by = {n["id"]: n for n in module["nodes"]}
    m = copy.deepcopy(structure)
    m["nodes"] = [copy.deepcopy(by[n["id"]]) for n in structure["nodes"]]
    expected = {
        14: "f32",
        19: "f32",
        20: "f32",
        21: "f32",
        22: "f32",
        23: "f16",
        29: "f16",
    }
    check(
        {i: n["dtype"] for i, n in enumerate(m["nodes"]) if "precision" in n}
        == expected,
        "resident chain requires explicit V/probability/PV/O/Z/RMS/final precision boundaries",
    )
    for i, n in enumerate(m["nodes"]):
        if i not in expected and n["op"] != "output":
            check(n.get("dtype") == "f16", "resident chain remaining storage is f16")
    # The sink's shape/type is inferred by structural verification, not author input.
    m["nodes"][-1].update(shape=structure["nodes"][-1]["shape"], dtype="f16")
    m.update(profile="mesh_input_attention_mixed.v1", precision_boundaries=facts)
    return m, structure
