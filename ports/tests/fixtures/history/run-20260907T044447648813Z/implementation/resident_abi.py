"""Declarative SDK I/O contract for resident algorithms, emitted with CSL."""

from mesh_cg import exports


def schema(s):
    power = s.get("solver") == "power"
    cap = s["max_iterations"]
    local = s["geometry"]["local_vec_sz"]
    lengths = dict(
        cg_rhs=local,
        cg_solution=local,
        cg_diagonal=local,
        cg_history=cap + 1,
        cg_scalars=3 * cap,
    )
    symbols = {
        name: dict(dtype=kind, length=length or lengths[name])
        for name, (kind, length) in exports(s).items()
    }
    symbols.update(
        hls_progress=dict(dtype="u16", length=11),
        hls_partial=dict(dtype="f32", length=2),
        y_local_buf=dict(dtype="f32", length=local),
    )
    fields = (
        dict(
            vector="cg_solution",
            reason="cg_reason",
            iterations="cg_iterations",
            norms="cg_history",
        )
        if power
        else dict(
            solution="cg_solution",
            reason="cg_reason",
            iterations="cg_iterations",
            residual_squared="cg_history",
            true_residual_norm="cg_true_norm",
        )
    )
    return dict(
        version="resident.sdk-abi.v1",
        launch="f_cg",
        vector_inputs=(
            [["cg_solution", 3]] if power else [["cg_rhs", 3], ["cg_solution", 4]]
        ),
        replicated_inputs=(
            [["cg_limit", 4, "u32"]]
            if power
            else [["cg_limit", 5, "u32"], ["cg_tolerances", 6, "f32"]]
        ),
        packed_inputs=(
            [["cg_diagonal", "diagonal", "f32"]]
            if s.get("preconditioner") == "jacobi"
            else []
        ),
        symbols=symbols,
        read_symbols=[
            name
            for name in symbols
            if name not in ("cg_rhs", "cg_limit", "cg_tolerances")
        ],
        result_fields=fields,
        vector_field="vector" if power else "solution",
        output_node_start=6 if power else 8,
        transport_order="COL_MAJOR",
        logical_vector_order="row-major PE ownership",
    )
