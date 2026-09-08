"""Explicit resident pencil placement, nearest-neighbor routes, resource ownership."""

import math
from frontend import check


def validate_resources(resources):
    # SDK 2.10.1/WSE3 compiler boundaries are preserved in resource-probes evidence.
    microthreads = resources["neighbor_receive_microthreads"] + [
        resources["send_microthread"],
        resources["initial_forward_microthread"],
        resources["final_output_microthread"],
    ]
    check(all(type(v) is int and 0 <= v < 8 for v in microthreads), "microthread range")
    check(len(set(microthreads)) == len(microthreads), "microthread ownership overlap")
    tasks = resources["local_tasks"]
    check(all(type(v) is int and 8 <= v < 31 for v in tasks), "local task range")
    check(len(set(tasks)) == len(tasks), "local task collision")


def plan(m, partitions=1):
    check(partitions == 1, "split-K is not a grid pass")
    op = m["nodes"][2]
    g = op["grid"]
    z = g["z"]
    coeff = m["nodes"][1]["shape"][1]
    nodes = []
    terms = None
    if op.get("vectorize"):
        from vectorize import ordered_terms

        terms = ordered_terms(op["body"], z)
    for x in range(g["x"]):
        for y in range(g["y"]):
            neighbors = {}
            for d, xx, yy in (
                ("west", x - 1, y),
                ("east", x + 1, y),
                ("south", x, y - 1),
                ("north", x, y + 1),
            ):
                if 0 <= xx < g["x"] and 0 <= yy < g["y"]:
                    neighbors[d] = f"p{xx}_{yy}"
            scratch = 4 * sum(
                math.prod(v)
                for k, v in op["body"]["arrays"].items()
                if k not in ("a", "b")
            )
            memory = {
                "neighborhood": 7 * z * 4,
                "owned_dsd_frame": (z + z % 2) * 4,
                "owned_receive_frame": (z + z % 2) * 4,
                "coefficients": coeff * 4,
                "output": (z + z % 2) * 4,
                "trace": z * g["steps"] * m["epochs"] * 4,
                "kernel_scratch": z * 4 if terms else scratch,
                "initial_forward": (g["y"] - y - 1) * (z + coeff + (z + coeff) % 2) * 4,
                "final_gather": (g["y"] - y) * (z + z % 2) * 4,
                "control_stack_reserve": 2048,
            }
            check(
                sum(memory.values()) <= 48 * 1024,
                "grid PE memory budget; reduce z/steps/epochs",
            )
            nodes.append(
                {
                    "id": f"p{x}_{y}",
                    "tile": [x, y],
                    "place": [4 + 3 * x, 4 + 3 * y],
                    "neighbors": neighbors,
                    "ingress_size": (g["y"] - y) * (z + coeff + (z + coeff) % 2),
                    "forward_size": (g["y"] - y - 1) * (z + coeff + (z + coeff) % 2),
                    "egress_size": (g["y"] - y) * (z + z % 2),
                    "collect_size": (g["y"] - y - 1) * (z + z % 2),
                    "memory": memory,
                    "resources": {
                        "neighbor_receive_microthreads": [
                            3 + i
                            for i, d in enumerate(("west", "east", "south", "north"))
                            if d in neighbors
                        ],
                        "send_microthread": 0,
                        "initial_forward_microthread": 1,
                        "final_output_microthread": 2,
                        "local_tasks": [8, 9, 10, 11, 12],
                        "initial_input_queue": 2,
                        "gather_input_queue": 7 if g["y"] - y - 1 else None,
                    },
                }
            )
    for node in nodes:
        validate_resources(node["resources"])
    return {
        "profile": "grid.v1",
        "vector_terms": terms,
        "numeric_lowering": (
            "ordered fmuls/fadds; no fusion or reassociation"
            if terms
            else "scalar body IR"
        ),
        "grid": g,
        "nodes": nodes,
        "coefficients": coeff,
        "epochs": m["epochs"],
        "input_bound": m["input_bound"],
        "host_input_size": z + coeff + (z + coeff) % 2,
        "wire_z": z + z % 2,
        "policy": "resident pencil; exchange old center with four neighbors; compute after all sends complete and full neighbor frames received; zero exterior; host only initial/final",
        "body": op["body"],
    }
