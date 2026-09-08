"""Lower scheduled frame actors to CSL; scalar transport is separately owned."""

from pathlib import Path
import shutil


def expr(e):
    if e[0] == "var":
        return "x"
    if e[0] == "const":
        return str(float(e[1]))
    if e[0] == "neg":
        return "(-" + expr(e[1]) + ")"
    return "(" + expr(e[1]) + e[0] + expr(e[2]) + ")"


def actor(n, by, epochs):
    sizes = n["wire_input_sizes"]
    na = sizes[0]
    nb = sizes[1] if len(sizes) > 1 else 0
    o = n["output_size"]
    wire = n["wire_output_size"]
    r, c = n["shape"]
    op = n["op"]
    ports = n["send_ports"]
    body = "for (@range(u16,L)) |i| { result[i]=a[i]; }"
    if op == "slice":
        sr, sc = by[n["inputs"][0]]["shape"]
        row = n["start"] if n["axis"] == 0 else 0
        col = n["start"] if n["axis"] == 1 else 0
        body = f"for (@range(u16,{r})) |i| {{ for (@range(u16,{c})) |j| {{ result[i*{c}+j]=a[(i+{row})*{sc}+j+{col}]; }} }}"
    if op == "kernel":
        from local_kernel import emit_kernel

        body, extra = emit_kernel(n["body"])
    else:
        extra = ""
    if op == "map":
        body = (
            "for (@range(u16,L)) |i| { const x=a[i]; result[i]="
            + expr(n["expr"])
            + "; }"
        )
    if op == "add":
        body = "for (@range(u16,L)) |i| { result[i]=a[i]+b[i]; }"
    if op == "accumulate":
        body = "for (@range(u16,L)) |i| { state[i]+=a[i]; result[i]=state[i]; }"
    if op == "transpose":
        body = f"for (@range(u16,{r})) |i| {{ for (@range(u16,{c})) |j| {{ result[i*{c}+j]=a[j*{r}+i]; }} }}"
    if op == "row_sum":
        k = by[n["inputs"][0]]["shape"][1]
        body = f"for (@range(u16,{r})) |i| {{ result[i]=0.0; for (@range(u16,{k})) |j| {{ result[i]+=a[i*{k}+j]; }} }}"
    if op == "matmul":
        k = by[n["inputs"][0]]["shape"][1]
        body = f"for (@range(u16,{r})) |i| {{ for (@range(u16,{c})) |j| {{ result[i*{c}+j]=0.0; for (@range(u16,{k})) |t| {{ result[i*{c}+j]+=a[i*{k}+t]*b[t*{c}+j]; }} }} }}"
    if wire > o:
        body += f"result[{o}]=0.0;"
    param_b = "param rx1:u16;" if nb else ""
    param_t = "param tx1:u16;" if ports == 2 else ""
    defs_b = (
        f"const iq1=@get_input_queue(3);const tid1=@get_data_task_id(iq1);var b=@zeros([{nb}]f32);var cb:u16=0;export var received_b:u32=0;"
        if nb
        else ""
    )
    defs_t = "const oq1=@get_output_queue(3);" if ports == 2 else ""
    state = f"export var state=@zeros([{o}]f32);" if op == "accumulate" else ""
    send = (
        "if(sent<O){comm.send(@bitcast(u32,result[sent]),oq0,done);}else{comm.send(@bitcast(u32,result[sent-O]),oq1,done);}"
        if ports == 2
        else "comm.send(@bitcast(u32,result[sent]),oq0,done);"
    )
    ready = f"ca=={na}" + (f" and cb=={nb}" if nb else "")
    recv_b = (
        f"""task receive1(data:u32) void {{ @assert(cb<{nb} and received_b<{nb}*EPOCHS); b[cb]=@bitcast(f32,data); cb+=1;received_b+=1;if(cb=={nb}){{@block(tid1);ready();}} }}"""
        if nb
        else ""
    )
    reset_b = "cb=0;@unblock(tid1);" if nb else ""
    init_b = (
        "@initialize_queue(iq1,.{.color=@get_color(rx1)});@bind_data_task(receive1,tid1);"
        if nb
        else ""
    )
    init_t = "@initialize_queue(oq1,.{.color=@get_color(tx1)});" if ports == 2 else ""
    return f"""// Middleware tensor actor {n["id"]}, source line {n["line"]}, op {op}.
param rx0:u16;param tx0:u16;{param_b}{param_t}
const L:u16={o};const O:u16={wire};const EPOCHS:u16={epochs};
{extra}
const math=@import_module("<math>");
const comm=@import_module("comm_runtime.csl");
const iq0=@get_input_queue(2);const oq0=@get_output_queue(2);
const tid0=@get_data_task_id(iq0);const done=@get_local_task_id(8);
{defs_b}{defs_t}{state}
var a=@zeros([{na}]f32);var result=@zeros([O]f32);var ca:u16=0;var sent:u16=0;
export var history=@zeros([L*EPOCHS]f32);
export var received_a:u32=0;export var produced:u32=0;export var completed:u32=0;
export var epochs:u16=0;export var inflight:u32=0;
fn transmit() void {{ {send} }}
fn ready() void {{ if({ready}){{
 @assert(inflight==0 and epochs<EPOCHS);inflight=1;
 {body}
 for (@range(u16,L)) |i| {{history[produced+i]=result[i];}}
 produced+=L;transmit();
}} }}
task receive0(data:u32) void {{ @assert(ca<{na} and received_a<{na}*EPOCHS); a[ca]=@bitcast(f32,data);ca+=1;received_a+=1;if(ca=={na}){{@block(tid0);ready();}} }}
{recv_b}
task complete() void {{
 comm.release();sent+=1;completed+=1;
 if(sent<{ports}*O){{transmit();}}else{{epochs+=1;inflight=0;ca=0;sent=0;{reset_b}@unblock(tid0);}}
}}
comptime {{
 @comptime_assert(@is_arch("wse3"));
 @initialize_queue(iq0,.{{.color=@get_color(rx0)}});@initialize_queue(oq0,.{{.color=@get_color(tx0)}});
 @bind_data_task(receive0,tid0);@bind_local_task(complete,done);{init_b}{init_t}
}}
"""


def generate(schedule, dest):
    if schedule.get("profile") == "mesh_projected_cache_ffn.v1":
        from mesh_projected_cache_ffn import generate as composed

        return composed(schedule, dest)

    if schedule.get("profile") == "mesh_projected_cache.v1":
        from mesh_projected_cache import generate as projected

        return projected(schedule, dest)
    if schedule.get("profile") == "mesh_input_attention_mixed.v1":
        from mesh_input_attention_mixed import generate as mixed_generate

        return mixed_generate(schedule, dest)

    if schedule.get("profile") == "mesh_attention_tail.v1":
        from mesh_attention_tail import generate as attention_tail_generate

        return attention_tail_generate(schedule, dest)
    if schedule.get("profile") == "mesh_prefill_tail.v1":
        from mesh_prefill_tail import generate as tail_generate

        return tail_generate(schedule, dest)
    if schedule.get("profile") == "mesh_cache_attention.v1":
        from mesh_cache_attention import generate as cache_generate

        return cache_generate(schedule, dest)
    if schedule.get("profile") == "mesh_batched_feed_forward.v1":
        from mesh_batched_feed_forward import generate as batch_ffn_generate

        return batch_ffn_generate(schedule, dest)
    if schedule.get("profile") == "mesh_feed_forward.v1":
        from mesh_feed_forward import generate as feed_forward_generate

        return feed_forward_generate(schedule, dest)
    if schedule.get("profile") == "mesh_projection_residual_rms.v1":
        from mesh_projection_residual_rms import generate as composition_generate

        return composition_generate(schedule, dest)
    if schedule.get("profile") == "mesh_mlp.v1":
        from mesh_mlp import generate as mlp_generate

        return mlp_generate(schedule, dest)
    if schedule.get("profile") == "mesh_attention.v1":
        from mesh_attention import generate as attention_generate

        return attention_generate(schedule, dest)
    if schedule.get("profile") == "mesh_score_softmax.v1":
        from mesh_score_softmax import generate as resident_generate

        return resident_generate(schedule, dest)
    if schedule.get("profile") == "mesh_device_matmul.v1":
        from mesh_device_matmul import generate as device_generate

        return device_generate(schedule, dest)
    if schedule.get("profile") == "mesh_score.v1":
        from mesh_score import generate as score_generate

        return score_generate(schedule, dest)
    if schedule.get("profile") == "mesh_pair_rotation.v1":
        from mesh_pair_rotation import generate as pair_generate

        return pair_generate(schedule, dest)
    if schedule.get("profile") == "mesh_swiglu.v1":
        from mesh_swiglu import generate as gating_generate

        return gating_generate(schedule, dest)

    if schedule.get("profile") == "mesh_normalized_fanout.v1":
        from mesh_normalized_fanout import generate as fanout_generate

        return fanout_generate(schedule, dest)
    if schedule.get("profile") == "mesh_normalized_matmul.v1":
        from mesh_normalized_matmul import generate as resident_generate

        return resident_generate(schedule, dest)
    if schedule.get("profile") == "mesh_softmax.v1":
        from mesh_softmax import generate as softmax_generate

        return softmax_generate(schedule, dest)
    if schedule.get("profile") == "mesh_batched_fanout.v1":
        from mesh_batched_fanout import generate as batched_fanout

        return batched_fanout(schedule, dest)
    if schedule.get("profile") == "mesh_batched_rms.v1":
        from mesh_batched_rms import generate as batched_generate

        return batched_generate(schedule, dest)
    if schedule.get("profile") == "mesh_rms.v1":
        from mesh_rms import generate as rms_generate

        return rms_generate(schedule, dest)
    if schedule.get("profile") == "mesh_fft.v1":
        from mesh_fft import generate as fft_generate

        return fft_generate(schedule, dest)
    if schedule.get("profile") == "mesh_grouped_gemv.v1":
        from mesh_grouped_gemv import generate as grouped_generate

        return grouped_generate(schedule, dest)
    if schedule.get("profile") == "mesh_twohop.v1":
        from mesh_twohop import generate as half_generate

        return half_generate(schedule, dest)
    if schedule.get("profile") in ("mesh_cg.v1", "mesh_power.v1"):
        from mesh_cg import generate as cg_generate

        return cg_generate(schedule, dest)
    if schedule.get("profile") == "mesh_reduction.v1":
        from mesh_reduction import generate as reduction_generate

        return reduction_generate(schedule, dest)
    if schedule.get("profile") == "mesh_spmv.v1":
        from mesh_spmv import generate as sparse_generate

        return sparse_generate(schedule, dest)
    if schedule.get("profile") == "mesh_qr.v1":
        from mesh_qr import generate as qr_generate

        return qr_generate(schedule, dest)
    if schedule.get("profile") == "mesh_lu.v1":
        from mesh_lu import generate as lu_generate

        return lu_generate(schedule, dest)
    if schedule.get("profile") == "mesh_cholesky.v1":
        from mesh_cholesky import generate as chol_generate

        return chol_generate(schedule, dest)
    if schedule.get("profile") == "mesh_cannon.v1":
        from mesh_cannon import generate as cannon_generate

        return cannon_generate(schedule, dest)
    if schedule.get("profile") == "mesh_gemm.v1":
        from mesh_gemm import generate as mesh_generate

        return mesh_generate(schedule, dest)
    if schedule.get("profile") == "mesh_gemv.v1":
        from mesh_gemv import generate as mesh_generate

        return mesh_generate(schedule, dest)
    if schedule.get("profile") == "grid.v1":
        from grid_backend import generate as grid_generate

        return grid_generate(schedule, dest)
    dest = Path(dest)
    by = {n["id"]: n for n in schedule["nodes"]}
    for n in schedule["nodes"]:
        (dest / (n["id"] + ".csl")).write_text(actor(n, by, schedule["epochs"]))
    shutil.copyfile(
        Path(__file__).parent / "runtime/comm_runtime.csl", dest / "comm_runtime.csl"
    )
