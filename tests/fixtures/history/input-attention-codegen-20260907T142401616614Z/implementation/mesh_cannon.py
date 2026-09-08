"""Cannon cyclic block contraction, retaining upstream skew and ring schedule."""

from pathlib import Path
from frontend import check
from mesh_common import verify_matmul, pack_tiles


def verify(module, epochs, bound):
    m = verify_matmul(module, epochs, bound)
    d = m["nodes"][2]["dataflow"]
    check(
        set(d)
        == {"rows", "cols", "exchange", "initial_align", "reduce", "fp", "compute"},
        "Cannon policy fields",
    )
    check(
        d["exchange"] == "cyclic"
        and d["initial_align"] == "host"
        and d["reduce"] == "local",
        "Cannon data movement",
    )
    m["profile"] = "mesh_cannon.v1"
    plan(m)
    return m


def plan(m, partitions=1):
    a, b, op, out = m["nodes"]
    d = op["dataflow"]
    p = d["rows"]
    n = a["shape"][0]
    check(
        partitions == 1 and type(p) is int and p in (4, 8) and d["cols"] == p,
        "Cannon even square4/8 mesh",
    )
    check(
        a["shape"] == b["shape"] == op["shape"] == [n, n] and n % p == 0,
        "Cannon square divisible matrices",
    )
    t = n // p
    memory = dict(
        matrix_buffers=16 * t * t,
        round_history=4 * p * t * t,
        witnesses=16 * p,
        timestamps_and_counts=12 * p + 64,
        sdk_code_stack_reserve=8192,
    )
    check(
        sum(memory.values()) <= 49152 and p * t * t <= 32767, "Cannon memory/DSD extent"
    )
    return dict(
        profile="mesh_cannon.v1",
        matrix_rows=n,
        matrix_k=n,
        matrix_cols=n,
        P=p,
        Mt=t,
        Kt=t,
        Nt=t,
        epochs=m["epochs"],
        compute=d["compute"],
        tile_order="row-major",
        memory_per_pe=memory,
        rounds=p,
        nodes=[
            dict(id=f"p{x}_{y}", tile=[x, y], place=[4 + x, 1 + y])
            for y in range(p)
            for x in range(p)
        ],
        stages=[
            "host permutation A[y,(x+y)%P],B[(x+y)%P,x]",
            "local accumulation for k_block=(x+y+round)%P",
            "parity-ordered left A shift and temporary ownership swap",
            "parity-ordered up B shift and temporary ownership swap",
        ],
        numeric_policy="explicit relaxed FMA; cyclic block K order",
        validation_policy="componentwise-f32-dot-v1; fixed accuracy reported separately",
        resources=dict(
            colors=list(range(6)),
            input_queues=[2, 3],
            output_queues=[2, 3],
            memcpy_queues=[0, 1],
            local_tasks=[],
            completion="synchronous fabric DSD transfers; parity order on even rings",
            buffers="A,B,temporary rotate; C retained; reset ownership every epoch",
        ),
    )


def pack(matrix, p, operand):
    import numpy as np

    check(operand in ("A", "B"), "Cannon operand")
    tiles = pack_tiles(matrix, p, p, "C")
    return np.asarray(
        [
            [
                tiles[y, (x + y) % p] if operand == "A" else tiles[(x + y) % p, x]
                for x in range(p)
            ]
            for y in range(p)
        ]
    )


def generate(s, dest):
    runtime = Path(__file__).parent / "runtime"
    code = (runtime / "cannon_pe.csl").read_text()
    if s["compute"] == "scalar":
        begin = code.index("fn compute_C() void {")
        end = code.index("\nfn send_A()", begin)
        code = code[:begin] + """fn compute_C() void {
 for(@range(i16,M_per_pe)) |i| {
  for(@range(i16,N_per_pe)) |k| {
   for(@range(i16,N_per_pe)) |j| {C[i*N_per_pe+j]+=A_ptr[i*N_per_pe+k]*B_ptr[k*N_per_pe+j];}
  }
 }
}
""" + code[end:]
    dest = Path(dest)
    (dest / "pe.csl").write_text(code)
    (dest / "layout.csl").write_text((runtime / "cannon_layout.csl").read_text())
