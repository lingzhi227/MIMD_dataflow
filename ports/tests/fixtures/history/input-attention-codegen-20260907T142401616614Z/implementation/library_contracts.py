"""Checked contracts for native CSL library adapters.

A contract is not automatic code selection. The caller must explicitly accept
the numerical policy and own the library's complete layout and resource scope.
"""

from frontend import check
from grid_ir import verify
from vectorize import ordered_terms


def sdk_stencil(module, block_size, *, allow_reassociation=False):
    m = verify(module, module["epochs"], module["input_bound"])
    check(
        allow_reassociation, "SDK stencil requires explicit FMA/reassociation consent"
    )
    g = m["nodes"][2]["grid"]
    z = g["z"]
    check(
        g["x"] > 1,
        "SDK benchmark adapter imports allreduce requiring width > 1; not a stencil limitation",
    )
    check(m["nodes"][1]["shape"] == [1, 7], "SDK stencil requires seven coefficients")
    terms = ordered_terms(m["nodes"][2]["body"], z)
    check(
        sorted(map(tuple, terms)) == [(i, i) for i in range(7)],
        "SDK stencil requires exactly one matching coefficient per neighborhood plane",
    )
    check(
        z >= 2 and type(block_size) is int and 2 <= block_size <= z,
        "SDK stencil requires z >= 2 and 2 <= block size <= z",
    )
    check(z * g["steps"] <= 32767, "SDK stencil history offset exceeds i16")
    # This adapter stores one epoch at a time; SDK memcpy/control overhead is
    # reserved conservatively, and compiler memory acceptance remains required.
    memory = {
        "resident_fields": 2 * z * 4,
        "neighbor_blocks": 4 * block_size * 4,
        "history": z * g["steps"] * 4,
        "coefficients": 7 * 4,
        "sdk_control_and_stack_reserve": 8192,
    }
    check(sum(memory.values()) <= 48 * 1024, "SDK adapter PE memory budget")
    return {
        "contract_version": 1,
        "library": "sdk-examples/stencil_3d_7pts/wse3",
        "source_commit": "4866cf330333446cb5e529e10f36be4600d1df29",
        "target": "SDK 2.10.1 / WSE3",
        "layout": "owns contiguous rectangle and internal local color configuration",
        "boundary": "zero exterior",
        "coefficient_permutation": [0, 1, 3, 2, 4, 5, 6],
        "numeric_policy": "FMA and reassociation explicitly enabled; not bitwise equivalence",
        "block_size": block_size,
        "memory_budget": memory,
        "resources": {
            "stencil_colors": list(range(8)),
            "stencil_local_tasks": [14, 15, 16],
            "receive_queues_and_microthreads": [4, 5, 6, 7],
            "send_queues": [4, 5, 6, 7],
            "send_microthread": 3,
            "dest_dsrs": [2, 3],
            "src0_dsrs": [2],
            "src1_dsrs": [2, 3],
            "memcpy_queues": [0, 1],
            "benchmark_allreduce_queue": 2,
            "benchmark_allreduce_color": 8,
            "benchmark_local_tasks": [13, 17, 18, 19, 20],
        },
        "lifetime": "x/y and coefficients owned until callback; next iteration starts only in callback",
        "integration_status": "executed native baseline adapter; production backend selection pending",
    }
