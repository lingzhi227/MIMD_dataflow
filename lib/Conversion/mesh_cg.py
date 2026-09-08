from source_tree import source_path, logical_name
"""Resident CG spatial plan and CSL composition over shared SDK-backed primitives."""

import copy, re
from pathlib import Path
from frontend import check
from mesh_spmv import (
    verify as sparse_verify,
    plan as sparse_plan,
    generate as sparse_generate,
    TEMPLATES as SPARSE_TEMPLATES,
)
from sparse_storage import Capacity, partition
from solver_reference import matrix_inputs, diagonal

ROOT = Path(__file__).resolve().parent
EXPORTS = {
    "cg_rhs": ("f32", None),
    "cg_solution": ("f32", None),
    "cg_limit": ("u32", 1),
    "cg_tolerances": ("f32", 2),
    "cg_reason": ("u32", 1),
    "cg_iterations": ("u32", 1),
    "cg_history": ("f32", None),
    "cg_true_norm": ("f32", 1),
    "cg_scalars": ("f32", None),
    "cg_progress": ("u16", 6),
    "cg_queue_last": ("u16", 2),
    "cg_timing": ("u16", 6),
}
TEMPLATES = {
    **SPARSE_TEMPLATES,
    "scalar_allreduce.csl": "scalar_allreduce.csl",
    "blas.csl": "sdk_blas.csl",
}


def sparse_module(m, vector_index=4, solve_index=7):
    solve = m["nodes"][solve_index]
    n = solve["result_type"]["dimension"]
    d = solve["dataflow"]
    inputs = copy.deepcopy([m["nodes"][i] for i in (0, 1, 2, vector_index)])
    op = dict(
        id="cg_sparse_operator",
        op="spmv_csc",
        shape=[n, 1],
        inputs=[x["id"] for x in inputs],
        dataflow={
            k: d[k]
            for k in (
                "rows",
                "cols",
                "storage",
                "exchange",
                "nnz_per_pe",
                "cols_per_pe",
                "rows_per_pe",
                "fp",
            )
        },
    )
    op["dataflow"]["reduce"] = "sparse_rows"
    out = dict(
        id="cg_sparse_sink",
        op="output",
        shape=None,
        inputs=[op["id"]],
        host="internal_spmv",
    )
    return sparse_verify(
        dict(m, nodes=inputs + [op, out]), m["epochs"], m["input_bound"]
    )


def resident_resources(spmv):
    return dict(
        spmv=spmv,
        collective_colors=[0, 7, 8, 9],
        collective_local_tasks=[8, 9, 22, 23],
        callback_task=10,
        collective_input_queues=[3, 6, 5, 7],
        collective_output_queues=[3, 6, 5, 7],
        ownership="single flight: SDK callbacks, checked drained queues and explicit recoloring; transpose borrows inactive north-partial storage",
    )


def plan(m, partitions=1):
    check(partitions == 1, "resident solver owns partitioning")
    t = m["nodes"][7]["result_type"]
    d = m["nodes"][7]["dataflow"]
    check(
        t["dimension"] == 512
        and d["rows"] == d["cols"] == 4
        and t["max_iterations"] <= 64,
        "initial resident CG lowering supports512 on4x4 with at most64 iterations",
    )
    if m["nodes"][7]["op"] == "bicgstab_csc":
        check(
            t["max_iterations"] <= 32,
            "BiCGStab resident history supports at most32 updates",
        )
    s = sparse_plan(sparse_module(m))
    local = s["geometry"]["local_vec_sz"]
    check(local * s["cols"] <= s["capacity"]["rows"], "CG transpose arena capacity")
    # Replace the sparse-only reserve with combined code/task/controller reserve.
    extra = 4 * local * 4 + 4 * (4 * t["max_iterations"] + 2) + 256
    memory = dict(
        sparse_data_estimate=s["estimated_bytes"] - 8192,
        solver_data_estimate=extra,
        combined_code_tasks_stack_reserve=32768,
    )
    check(sum(memory.values()) <= 49152, "combined CG static estimate exceeds48KiB")
    s.update(
        profile="mesh_cg.v1",
        max_iterations=t["max_iterations"],
        memory_per_pe=memory,
        estimated_bytes=sum(memory.values()),
        resources=resident_resources(s["resources"]),
        stages=[
            "stable RHS norm",
            "initial Ax and residual",
            "resident transpose/SpMV/dot/FMA recurrence",
            "device convergence and iteration limit",
            "final original-operator residual norm",
            "structured result transport",
        ],
    )
    if m["nodes"][7]["op"] == "pcg_csc":
        s["preconditioner"] = "jacobi"
        extra = 8 * local + 4 * t["max_iterations"]
        s["memory_per_pe"]["jacobi_state_estimate"] = extra
        s["estimated_bytes"] += extra
        check(s["estimated_bytes"] <= 49152, "Jacobi combined estimate exceeds48KiB")
    if m["nodes"][7]["op"] == "bicgstab_csc":
        s["solver"] = "bicgstab"
        extra = 12 * local + 20 * t["max_iterations"] + 64
        s["memory_per_pe"]["bicgstab_state_estimate"] = extra
        s["estimated_bytes"] += extra
        check(s["estimated_bytes"] <= 49152, "BiCGStab combined estimate exceeds48KiB")
    return s


def packing(m, batches):
    s = plan(m)
    packs = []
    for batch in batches:
        a = matrix_inputs(m, batch)[0]
        p = partition(a, s["rows"], s["cols"], Capacity(**s["capacity"]))
        if s.get("preconditioner") == "jacobi":
            # Copy original diagonal values into resident vector ownership. No host reciprocal.
            d = diagonal(a)
            l = s["geometry"]["local_vec_sz"]
            p["diagonal"] = [
                [
                    d[(y * s["cols"] + x) * l : (y * s["cols"] + x + 1) * l]
                    for x in range(s["cols"])
                ]
                for y in range(s["rows"])
            ]
        packs.append(p)
    return packs


def exports(s):
    if s.get("solver") == "power":
        return {
            **{
                k: v
                for k, v in EXPORTS.items()
                if k not in ("cg_rhs", "cg_tolerances", "cg_true_norm", "cg_scalars")
            },
            "power_attempts": ("u16", 1),
            "power_inverse": ("f32", s["max_iterations"]),
        }
    if s.get("solver") == "bicgstab":
        cap = s["max_iterations"]
        return dict(
            EXPORTS,
            bi_products=("f32", 3 * cap),
            bi_rhos=("f32", cap),
            bi_ss=("f32", cap),
            bi_progress=("u16", 9),
            bi_failure=("u16", 1),
            bi_early=("u16", 1),
        )
    return dict(
        EXPORTS,
        **(
            {"cg_diagonal": ("f32", None), "cg_weights": ("f32", s["max_iterations"])}
            if s.get("preconditioner") == "jacobi"
            else {}
        ),
    )


def generate(s, dest):
    dest = Path(dest)
    sparse_generate(s, dest)
    for name, template in TEMPLATES.items():
        (dest / name).write_text((source_path('runtime') / template).read_text())
    layout = (dest / "layout.csl").read_text()
    layout = layout.replace(
        "layout {",
        'param max_iterations:u16;\nconst c2d=@import_module("<collectives_2d/params>");\nlayout {',
        1,
    )
    layout = layout.replace(
        ".spmvParams = spmvParams,",
        ".spmvParams = spmvParams,\n.cg_max_iterations=max_iterations,\n.c2dParams=c2d.get_params(pcol_id,prow_id,.{.x_colors=.{@get_color(0),@get_color(7)},.x_entrypoints=.{@get_local_task_id(8),@get_local_task_id(9)},.y_colors=.{@get_color(8),@get_color(9)},.y_entrypoints=.{@get_local_task_id(22),@get_local_task_id(23)}}),",
    )
    if s.get("preconditioner") == "jacobi":
        layout = layout.replace(
            ".cg_max_iterations=max_iterations,",
            ".cg_max_iterations=max_iterations,.cg_jacobi=true,",
        )
    if s.get("solver") == "power":
        layout = layout.replace(
            ".cg_max_iterations=max_iterations,",
            ".cg_max_iterations=max_iterations,.solver_kind=1,",
        )
    end = layout.rindex("}")
    declarations = "\n".join(
        f'@export_name("{name}",[*]{kind},true);'
        for name, (kind, length) in exports(s).items()
    )
    layout = (
        layout[:end]
        + declarations
        + '\n@export_name("f_cg",fn()void);\n'
        + layout[end:]
    )
    kernel = (
        (dest / "kernel.csl")
        .read_text()
        .replace(
            ".f_callback = sys_mod.unblock_cmd_stream,",
            ".f_callback = cg_resume, .initialize_queues=false,",
        )
    )
    # Retain the legacy diagnostic exports; only f_cg is used by this profile's host ABI.
    kernel += "\n" + (source_path('runtime/solver_common.csl')).read_text()
    controller = {
        "bicgstab": "bicgstab_controller.csl",
        "power": "power_controller.csl",
    }.get(s.get("solver"), "cg_controller.csl")
    kernel += "\n" + (source_path('runtime') / controller).read_text()
    (dest / "layout.csl").write_text(layout)
    (dest / "kernel.csl").write_text(kernel)
    import json
    from resident_abi import schema

    (dest / "host-abi.json").write_text(json.dumps(schema(s), indent=2) + "\n")
