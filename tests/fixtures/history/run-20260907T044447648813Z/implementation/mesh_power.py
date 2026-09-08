"""Power dataflow plan built on shared resident CSC/collective infrastructure."""

from pathlib import Path
from frontend import check
from mesh_spmv import plan as sparse_plan
from mesh_cg import sparse_module, resident_resources, generate
from sparse_storage import Capacity, partition
from power_ir import inputs


def plan(m, partitions=1):
    t = m["nodes"][5]["result_type"]
    d = m["nodes"][5]["dataflow"]
    check(
        partitions == 1 and t["dimension"] == 512 and d["rows"] == d["cols"] == 4,
        "power lowering512 on4x4",
    )
    s = sparse_plan(sparse_module(m, 3, 5))
    l = s["geometry"]["local_vec_sz"]
    check(l * s["cols"] <= s["capacity"]["rows"], "power transpose workspace")
    memory = dict(
        sparse_data_estimate=s["estimated_bytes"] - 8192,
        shared_vector_state_estimate=16 * l + 8 * t["max_iterations"] + 256,
        combined_code_tasks_stack_reserve=32768,
    )
    check(sum(memory.values()) <= 49152, "power combined static estimate")
    s.update(
        profile="mesh_power.v1",
        solver="power",
        max_iterations=t["max_iterations"],
        resources=resident_resources(s["resources"]),
        memory_per_pe=memory,
        estimated_bytes=sum(memory.values()),
        stages=[
            "requested fixed-step budget",
            "resident transpose and CSC SpMV",
            "stable global norm",
            "DSD normalization or zero-norm exit",
            "typed completion record",
        ],
    )
    return s


def packing(m, batches):
    s = plan(m)
    return [
        partition(inputs(m, b)[0], s["rows"], s["cols"], Capacity(**s["capacity"]))
        for b in batches
    ]
