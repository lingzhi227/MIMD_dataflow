"""CSL resident-grid state machine; kernel arithmetic comes from typed body IR."""

from pathlib import Path
import shutil
from local_kernel import emit_kernel

DIRECTIONS = ("west", "east", "south", "north")


def actor(n, s):
    z = s["grid"]["z"]
    steps = s["grid"]["steps"]
    epochs = s["epochs"]
    c = s["coefficients"]
    wire = s["wire_z"]
    ins = s["host_input_size"]
    dirs = [d for d in DIRECTIONS if d in n["neighbors"]]
    body, extra = emit_kernel(s["body"])
    if s.get("vector_terms"):
        from vectorize import emit

        body, extra = emit(s["vector_terms"], z)
    extra += f"\nconst center_dsd=@get_dsd(mem1d_dsd,.{{.tensor_access=|i|{{{z}}}->a[{6*z}+i]}});const result_dsd=@get_dsd(mem1d_dsd,.{{.tensor_access=|i|{{{z}}}->result[i]}});const trace_dsd=@get_dsd(mem1d_dsd,.{{.tensor_access=|i|{{{z}}}->history[i]}});"
    local_halo = f"a[{4*z}]=0.0;a[{6*z-1}]=0.0;"
    if z > 1:
        for name, offset in [
            ("bottom_dst", 4 * z + 1),
            ("top_dst", 5 * z),
            ("bottom_src", 6 * z),
            ("top_src", 6 * z + 1),
        ]:
            extra += f"const {name}=@get_dsd(mem1d_dsd,.{{.tensor_access=|i|{{{z-1}}}->a[{offset}+i]}});"
        local_halo += "@fmovs(bottom_dst,bottom_src);@fmovs(top_dst,top_src);"
    ingress = n["ingress_size"]
    forward = n["forward_size"]
    egress = n["egress_size"]
    collect = n["collect_size"]
    io_defs = f"var gather=@zeros([{egress}]f32);const gather_dsd=@get_dsd(mem1d_dsd,.{{.tensor_access=|i|{{{egress}}}->gather[i]}});const gather_head=@get_dsd(mem1d_dsd,.{{.tensor_access=|i|{{{z}}}->gather[i]}});const host_out=@get_dsd(fabout_dsd,.{{.extent={egress},.output_queue=oq_host}});"
    io_bindings = ""
    io_tasks = ""
    forward_start = ""
    if forward:
        io_defs += f"param tx_init:u16;const oq_init=@get_output_queue(7);var init_buffer=@zeros([{forward}]f32);const init_mem=@get_dsd(mem1d_dsd,.{{.tensor_access=|i|{{{forward}}}->init_buffer[i]}});const init_out=@get_dsd(fabout_dsd,.{{.extent={forward},.output_queue=oq_init}});"
        forward_start = "init_busy=true;@fmovs(init_out,init_mem,.{.async=true,.activate=init_done,.ut_id=@get_ut_id(1)});"
        io_bindings += "@initialize_queue(oq_init,.{.color=@get_color(tx_init)});"
    if collect:
        io_defs += "param rx_collect:u16;const iq_collect=@get_input_queue(7);const tid_collect=@get_data_task_id(iq_collect);"
        io_tasks = f"task collect_input(data:u32) void {{@assert(collect_count<{collect});gather[W+collect_count]=@bitcast(f32,data);collect_count+=1;collected+=1;if(collect_count=={collect}){{@block(tid_collect);maybe_output();}}}}"
        io_bindings += "@initialize_queue(iq_collect,.{.color=@get_color(rx_collect)});@bind_data_task(collect_input,tid_collect);"
    params = []
    defs = []
    tasks = []
    bindings = []
    for d in dirs:
        index = DIRECTIONS.index(d)
        q = 3 + index
        params.append(f"param rx_{d}:u16;param tx_{d}:u16;")
        defs.append(
            f"const iq_{d}=@get_input_queue({q});const oq_{d}=@get_output_queue({q});const tid_{d}=@get_data_task_id(iq_{d});var count_{d}:u16=0;"
        )
        bindings.append(
            f"@initialize_queue(iq_{d},.{{.color=@get_color(rx_{d})}});@initialize_queue(oq_{d},.{{.color=@get_color(tx_{d})}});"
        )
    tx = "".join(
        ("if" if i == 0 else "else if") + f"(send_dir=={i}){{comm.send(oq_{d},done);}}"
        for i, d in enumerate(dirs)
    )
    rx = "".join(
        ("if" if i == 0 else "else if")
        + f"(recv_dir=={i}){{comm.post_receive(iq_{d},received_id);}}"
        for i, d in enumerate(dirs)
    )
    release_rx = "".join(
        ("if" if i == 0 else "else if")
        + f"(recv_dir=={i}){{comm.release_receive(&a,@as(i16,{DIRECTIONS.index(d)*z}),Z);}}"
        for i, d in enumerate(dirs)
    )
    return f"""// Resident grid PE {n['id']}. No host intervention between steps.
param rx_host:u16;param tx_host:u16;{''.join(params)}
const Z:u16={z};const W:u16={wire};const STEPS:u16={steps};const EPOCHS:u16={epochs};
const math=@import_module("<math>");const comm=@import_module("frame_runtime.csl",.{{.width=W}});
const iq_host=@get_input_queue(2);const oq_host=@get_output_queue(2);const tid_host=@get_data_task_id(iq_host);
const done=@get_local_task_id(8);const compute_id=@get_local_task_id(9);const init_done=@get_local_task_id(10);const output_done=@get_local_task_id(11);const received_id=@get_local_task_id(12);
var a=@zeros([{7*z}]f32);var b=@zeros([{c}]f32);var result=@zeros([W]f32);
{extra}
{io_defs}
{''.join(defs)}
export var history=@zeros([{epochs*steps*z}]f32);
export var epochs:u16=0;export var step:u16=0;export var received:u32=0;export var sent:u32=0;export var host_received:u32=0;export var host_sent:u32=0;
export var forwarded:u32=0;export var collected:u32=0;
var loaded:u16=0;var send_dir:u16=0;var send_word:u16=0;var active:bool=false;var sent_all:bool=false;var recv_dir:u16=0;var recv_all:bool=false;var finished:bool=false;var init_busy:bool=false;var collect_count:u16=0;
fn ready() void {{if(active and sent_all and recv_all){{active=false;@activate(compute_id);}}}}
fn transmit() void {{
 {tx}
}}
fn fill_frame() void {{comm.load(&a,@as(i16,6*Z),Z);}}
fn receive_next() void {{{rx}}}
fn begin_exchange() void {{fill_frame();send_dir=0;send_word=0;recv_dir=0;recv_all=false;sent_all=false;active=true;
 {"recv_all=true;" if not dirs else "receive_next();"}
 {'sent_all=true;ready();' if not dirs else 'transmit();'}
}}
task load(data:u32) void {{
 @assert(loaded<{ingress} and epochs<EPOCHS);
 if(loaded<Z){{a[6*Z+loaded]=@bitcast(f32,data);}}else if(loaded<Z+{c}){{b[loaded-Z]=@bitcast(f32,data);}}else if(loaded<{ins}){{@assert(data==0);}}{'else{init_buffer[loaded-'+str(ins)+']=@bitcast(f32,data);}' if forward else ''}
 loaded+=1;host_received+=1;if(loaded=={ingress}){{@block(tid_host);step=0;{forward_start}begin_exchange();}}
}}
{''.join(tasks)}
{io_tasks}
fn maybe_output() void {{if(finished and collect_count=={collect}){{finished=false;@fmovs(host_out,gather_dsd,.{{.async=true,.activate=output_done,.ut_id=@get_ut_id(2)}});}}}}
task initial_sent() void {{@assert(init_busy);init_busy=false;forwarded+={forward};}}
task output_sent() void {{@assert(!init_busy);host_sent+={egress};epochs+=1;loaded=0;collect_count=0;{'@unblock(tid_collect);' if collect else ''}@unblock(tid_host);}}
task compute() void {{
 @assert(step<STEPS);
 {local_halo}
 {body}
 @fmovs(center_dsd,result_dsd);
 const trace_at=@increment_dsd_offset(trace_dsd,@as(i16,(epochs*STEPS+step)*Z),f32);
 @fmovs(trace_at,result_dsd);
 step+=1;if(step<STEPS){{begin_exchange();}}else{{@fmovs(gather_head,center_dsd);if(W>Z){{gather[Z]=0.0;}}finished=true;maybe_output();}}
}}
task received_frame() void {{{release_rx}received+=W;recv_dir+=1;if(recv_dir<{len(dirs)}){{receive_next();}}else{{recv_all=true;ready();}}}}
task complete() void {{
 comm.release();
 sent+=W;send_dir+=1;if(send_dir<{len(dirs)}){{transmit();}}else{{sent_all=true;ready();}}
}}
comptime{{
 @comptime_assert(@is_arch("wse3"));@initialize_queue(iq_host,.{{.color=@get_color(rx_host)}});@initialize_queue(oq_host,.{{.color=@get_color(tx_host)}});
 @bind_data_task(load,tid_host);@bind_local_task(complete,done);@bind_local_task(compute,compute_id);@bind_local_task(initial_sent,init_done);@bind_local_task(output_sent,output_done);@bind_local_task(received_frame,received_id);{io_bindings}{''.join(bindings)}
}}
"""


def generate(s, dest):
    dest = Path(dest)
    for n in s["nodes"]:
        (dest / (n["id"] + ".csl")).write_text(actor(n, s))
    shutil.copyfile(
        Path(__file__).parent / "runtime/frame_runtime.csl", dest / "frame_runtime.csl"
    )
