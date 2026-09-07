"""Schedule normalization and owned-buffer/resource planning, independent of CSL."""

import copy, math
from frontend import check


def optimize(m, enabled):
    m = copy.deepcopy(m)
    log = []
    if enabled:
        for n in list(m["nodes"]):
            if n["op"] == "map" and n["expr"] == ["var", "x"] and "place" not in n:
                for consumer in m["nodes"]:
                    consumer["inputs"] = [
                        n["inputs"][0] if x == n["id"] else x
                        for x in consumer["inputs"]
                    ]
                m["nodes"].remove(n)
                log.append(
                    {
                        "pass": "identity",
                        "removed": n["id"],
                        "reason": "pure identity; no hard placement or state effect",
                    }
                )
    m["rewrites"] = log
    return m


def plan(m, partitions=1):
    if m.get("profile") == "mesh_projection_residual_rms.v1":
        from mesh_projection_residual_rms import plan as composition_plan

        return composition_plan(m, partitions)
    if m.get("profile") == "mesh_mlp.v1":
        from mesh_mlp import plan as mlp_plan

        return mlp_plan(m, partitions)
    if m.get("profile") == "mesh_attention.v1":
        from mesh_attention import plan as attention_plan

        return attention_plan(m, partitions)
    if m.get("profile") == "mesh_score_softmax.v1":
        from mesh_score_softmax import plan as resident_plan

        return resident_plan(m, partitions)
    if m.get("profile") == "mesh_device_matmul.v1":
        from mesh_device_matmul import plan as device_plan

        return device_plan(m, partitions)
    if m.get("profile") == "mesh_score.v1":
        from mesh_score import plan as score_plan

        return score_plan(m, partitions)
    if m.get("profile") == "mesh_pair_rotation.v1":
        from mesh_pair_rotation import plan as pair_plan

        return pair_plan(m, partitions)
    if m.get("profile") == "mesh_swiglu.v1":
        from mesh_swiglu import plan as gating_plan

        return gating_plan(m, partitions)

    if m.get("profile") == "mesh_normalized_fanout.v1":
        from mesh_normalized_fanout import plan as fanout_plan

        return fanout_plan(m, partitions)
    if m.get("profile") == "mesh_normalized_matmul.v1":
        from mesh_normalized_matmul import plan as resident_plan

        return resident_plan(m, partitions)

    if m.get("profile") == "mesh_softmax.v1":
        from mesh_softmax import plan as softmax_plan

        return softmax_plan(m, partitions)
    if m.get("profile") == "mesh_rms.v1":
        from mesh_rms import plan as rms_plan

        return rms_plan(m, partitions)
    if m.get("profile") == "mesh_fft.v1":
        from mesh_fft import plan as fft_plan

        return fft_plan(m, partitions)
    if m.get("profile") == "mesh_grouped_gemv.v1":
        from mesh_grouped_gemv import plan as grouped_plan

        return grouped_plan(m, partitions)
    if m.get("profile") == "mesh_twohop.v1":
        from mesh_twohop import plan as half_plan

        return half_plan(m, partitions)
    if m.get("profile") == "mesh_power.v1":
        from mesh_power import plan as power_plan

        return power_plan(m, partitions)
    if m.get("profile") == "mesh_cg.v1":
        from mesh_cg import plan as cg_plan

        return cg_plan(m, partitions)
    if m.get("profile") == "mesh_reduction.v1":
        from mesh_reduction import plan as reduction_plan

        return reduction_plan(m, partitions)
    if m.get("profile") == "mesh_spmv.v1":
        from mesh_spmv import plan as sparse_plan

        return sparse_plan(m, partitions)
    if m.get("profile") == "mesh_qr.v1":
        from mesh_qr import plan as qr_plan

        return qr_plan(m, partitions)
    if m.get("profile") == "mesh_lu.v1":
        from mesh_lu import plan as lu_plan

        return lu_plan(m, partitions)
    if m.get("profile") == "mesh_cholesky.v1":
        from mesh_cholesky import plan as chol_plan

        return chol_plan(m, partitions)
    if m.get("profile") == "mesh_cannon.v1":
        from mesh_cannon import plan as cannon_plan

        return cannon_plan(m, partitions)
    if m.get("profile") == "mesh_gemm.v1":
        from mesh_gemm import plan as mesh_plan

        return mesh_plan(m, partitions)
    if m.get("profile") == "mesh_gemv.v1":
        from mesh_gemv import plan as mesh_plan

        return mesh_plan(m, partitions)
    if m.get("profile") == "grid.v1":
        from grid_plan import plan as grid_plan

        return grid_plan(m, partitions)
    check(type(partitions) is int and 1 <= partitions <= 4, "matmul partitions 1..4")
    nodes = []
    serial = 0
    original = {n["id"]: n for n in m["nodes"]}
    lowerings = []
    for n in copy.deepcopy(m["nodes"]):
        if n["op"] != "matmul" or partitions == 1:
            nodes.append(n)
            continue
        rows, k = original[n["inputs"][0]]["shape"]
        cols = n["shape"][1]
        check(partitions <= k, "more partitions than reduction elements")
        partials = []
        for part in range(partitions):
            start = part * k // partitions
            stop = (part + 1) * k // partitions
            count = stop - start
            a = {
                "id": "__mw_aslice" + str(serial),
                "op": "slice",
                "shape": [rows, count],
                "inputs": [n["inputs"][0]],
                "axis": 1,
                "start": start,
                "interval": original[n["inputs"][0]]["interval"],
                "line": n["line"],
            }
            b = {
                "id": "__mw_bslice" + str(serial),
                "op": "slice",
                "shape": [count, cols],
                "inputs": [n["inputs"][1]],
                "axis": 0,
                "start": start,
                "interval": original[n["inputs"][1]]["interval"],
                "line": n["line"],
            }
            product = {
                "id": "__mw_partial" + str(serial),
                "op": "matmul",
                "shape": n["shape"],
                "inputs": [a["id"], b["id"]],
                "interval": n["interval"],
                "line": n["line"],
            }
            serial += 1
            nodes.extend([a, b, product])
            partials.append(product["id"])
        # Ordered partial reduction; integer interval proof allows reassociation.
        running = partials[0]
        for part, next_id in enumerate(partials[1:], 1):
            last = part == len(partials) - 1
            add = copy.deepcopy(n)
            add.update(
                id=n["id"] if last else "__mw_reduce" + str(serial),
                op="add",
                inputs=[running, next_id],
            )
            serial += 1
            if not last:
                add.pop("place", None)
            nodes.append(add)
            running = add["id"]
        lowerings.append(
            {
                "operation": n["id"],
                "kind": "split_k",
                "partitions": partitions,
                "partials": partials,
                "tail_policy": "explicit uneven slices",
                "reduction": "ordered partial sums; explicit float reassociation with tolerance",
            }
        )
    # Bound all generated actors to two send ports; one frame of capacity per edge.
    for n in list(nodes):
        uses = [
            (c, i) for c in nodes for i, x in enumerate(c["inputs"]) if x == n["id"]
        ]
        parent = n["id"]
        while len(uses) > 2:
            fork = {
                "id": "__mw_fork" + str(serial),
                "op": "fork",
                "inputs": [parent],
                "shape": n["shape"],
                "interval": n["interval"],
                "line": n["line"],
            }
            serial += 1
            nodes.append(fork)
            # Keep one consumer on parent; route remaining consumers through the fork.
            uses = uses[1:]
            for c, i in uses:
                c["inputs"][i] = fork["id"]
            parent = fork["id"]
    known = {n["id"]: n for n in nodes}
    done = set()
    ordered = []
    while len(ordered) < len(nodes):
        ready = [
            n
            for n in nodes
            if n["id"] not in done and all(x in done for x in n["inputs"])
        ]
        check(ready, "cyclic schedule")
        ordered.extend(ready)
        done.update(n["id"] for n in ready)
    occupied = set()
    explicit = {tuple(n["place"]) for n in nodes if "place" in n}
    check(
        len(explicit) == sum("place" in n for n in nodes), "overlapping hard placements"
    )
    for i, n in enumerate(ordered):
        if "place" not in n:
            candidates = ((4 + 3 * (k % 8), 4 + 3 * (k // 8)) for k in range(64))
            n["place"] = list(
                next((p for p in candidates if p not in occupied | explicit), (-1, -1))
            )
        p = tuple(n["place"])
        check(
            len(p) == 2 and all(type(x) is int and 3 <= x <= 60 for x in p),
            "invalid placement or exhausted plan",
        )
        check(p not in occupied, "placement collision")
        occupied.add(p)
        n["input_sizes"] = [math.prod(known[x]["shape"]) for x in n["inputs"]] or [
            math.prod(n["shape"])
        ]
        n["output_size"] = math.prod(n["shape"])
        n["wire_input_sizes"] = [x + x % 2 for x in n["input_sizes"]]
        n["wire_output_size"] = n["output_size"] + n["output_size"] % 2
        n["consumers"] = [
            {"node": c["id"], "input": j}
            for c in nodes
            for j, x in enumerate(c["inputs"])
            if x == n["id"]
        ]
        check(len(n["input_sizes"]) <= 2 and len(n["consumers"]) <= 2, "port budget")
        n["send_ports"] = max(1, len(n["consumers"]))
        sizes = n["input_sizes"]
        out = n["output_size"]
        state = out if n["op"] == "accumulate" else 0
        n["memory"] = {
            "frame_inputs": 4 * sum(n["wire_input_sizes"]),
            "kernel_scratch": 4
            * sum(
                math.prod(shape)
                for key, shape in n.get("body", {}).get("arrays", {}).items()
                if key not in ("a", "b")
            ),
            "frame_output": 4 * n["wire_output_size"],
            "resident_state": 4 * state,
            "diagnostic_history": 4 * out * m["epochs"],
            "transport_owned_word": 4,
            "control_and_stack_reserve": 1280,
        }
        n["memory"]["planned_total"] = sum(n["memory"].values())
        check(n["memory"]["planned_total"] <= 48 * 1024, "managed memory budget")
        n["resources"] = {
            "input_queues": list(range(2, 2 + len(sizes))),
            "output_queues": list(range(2, 2 + n["send_ports"])),
            "completion_task": 8,
        }
    return {
        "version": "middleware.schedule.v1",
        "matmul_partitions": partitions,
        "lowerings": lowerings,
        "epochs": m["epochs"],
        "input_bound": m["input_bound"],
        "nodes": ordered,
        "policy": "full-frame per edge; all inputs ready before compute; sequential owned sends; one host epoch in flight",
        "memory_scope": "declared data plus conservative reserve; SDK compilation remains required",
        "dependencies": [
            {
                "producer": x,
                "consumer": n["id"],
                "input": i,
                "source_released": "local send completion",
                "destination_ready": "complete frame received",
                "capacity": math.prod(known[x]["shape"]),
            }
            for n in ordered
            for i, x in enumerate(n["inputs"])
        ],
    }
