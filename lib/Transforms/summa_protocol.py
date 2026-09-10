"""Extract the per-PE paired-completion obligation of an existing SUMMA plan.

Collective completion is a trusted boundary: the SDK must have filled the local
panel and finished accessing it before calling the specified completion task.
The actual SDK implements the join with activate/unblock; this abstraction uses
two one-shot events per round and does not model the task-bit implementation.
"""
from event_protocol import Op, Protocol


def from_plan(plan, *, omit_y_wait=False, early_reuse=False):
    if plan.get("profile") != "mesh_gemm.v1":
        raise ValueError("Expected an existing SUMMA plan")
    rounds = plan["rounds"]
    if type(rounds) is not int or not 1 <= rounds <= 8:
        raise ValueError("Finite profile supports 1..8 rounds")
    if plan["stages"][2]["after"] != ["broadcast_A", "broadcast_B"]:
        raise ValueError("SUMMA compute must join both broadcasts")
    actors = []
    for axis, buffer in (("x", "A"), ("y", "B")):
        ops = []
        for r in range(rounds):
            if r and not early_reuse:
                ops.append(Op("wait", f"computed:{r-1}"))
            ops.extend([Op("write_begin", buffer, r, "dma"),
                        Op("write_end", buffer, r, "dma"),
                        Op("signal", f"{axis}:{r}")])
        actors.append((axis, tuple(ops)))
    compute = []
    for r in range(rounds):
        compute.append(Op("wait", f"x:{r}"))
        if not omit_y_wait:
            compute.append(Op("wait", f"y:{r}"))
        compute.extend([Op("borrow", "A", r, "compute"), Op("borrow", "B", r, "compute"),
                        Op("release", "A", r, "compute"), Op("release", "B", r, "compute"),
                        Op("signal", f"computed:{r}")])
    actors.append(("compute", tuple(compute)))
    return Protocol(tuple(actors), ("A", "B"))


def mismatched_stream_order(capacity=1):
    # The data graph is a DAG with two parallel edges P -> C. Blocking writes
    # add a backwards dependency. Producer and consumer agree on token counts.
    return Protocol((("producer", (Op("put", "a"), Op("put", "a"), Op("put", "b"))),
                     ("consumer", (Op("get", "b"), Op("get", "a"), Op("get", "a")))),
                    fifos=(("a", capacity, 0), ("b", 1, 0)))
