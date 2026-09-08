"""Executed signed/positive witnesses for both levels of the Decode tree."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)

import json,sys,shutil,datetime
from pathlib import Path
import numpy as np
from probe_runtime import execute,mesh_half_worker,read,sha
ROOT = repository_root(__file__)
sys.path[:0]=[str(ROOT/'lib')]

def prepare(g):
    from grouped_collective_csl import generate
    from decode_grouped_reference import grouped
    root=ROOT/'validation/evidence'/('decode-grouped-order-g'+str(g)+'-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));root.mkdir()
    generate(root)
    text='''param memcpy_params;param groups:i16;
const P:i16=8;const size:i16=P/groups;
const sys=@import_module("<memcpy/memcpy>",memcpy_params);
const pos=@import_module("<layout>");const config=@import_module("<tile_config>");
const comm=@import_module("axis_grouped_reduce_dynamic.csl",.{.P=P,.bsz=8,.pe_num_p_group=size,.root_1st_phase=size/2,.root_2nd_phase=(groups/2)*size+size/2,.reduce_1st_color_0=@get_color(8),.reduce_1st_color_1=@get_color(7),.reduce_2nd_color_0=@get_color(6),.reduce_2nd_color_1=@get_color(5),.broadcast_color=@get_color(9)});
var A=@zeros([2]f16);var B=@zeros([8]f16);var C=@zeros([2]f16);var progress=@zeros([1]u16);var queues=@zeros([2]u16);
var py:i16=0;
fn init_task() void {const px=@as(i16,pos.get_x_coord());py=@as(i16,pos.get_y_coord());comm.init(px,py,px/size,px%size,py/size,py%size);sys.unblock_cmd_stream();}
fn hls_main() void {
 comm.set_extent(2);comm.all_reduce_bsz(py,py/size,py%size,&A);
 comm.set_extent(8);comm.all_reduce_bsz(py,py/size,py%size,&B);
 comm.set_extent(2);comm.all_reduce_bsz(py,py/size,py%size,&C);
 const qi=config.input_queue_status.get();const qo=config.output_queue_status.get();queues[0]=@as(u16,qi.empty);queues[1]=@as(u16,qo.empty);
 progress[0]+=1;sys.unblock_cmd_stream();
}
var ap:[*]f16=&A;var bp:[*]f16=&B;var cp:[*]f16=&C;var pp:[*]u16=&progress;var qp:[*]u16=&queues;
comptime {@export_symbol(ap,"A");@export_symbol(bp,"B");@export_symbol(cp,"C");@export_symbol(pp,"progress");@export_symbol(qp,"queues");@export_symbol(init_task);@export_symbol(hls_main);}
'''
    (root/'pe.csl').write_text(text)
    (root/'layout.csl').write_text('''param groups:i16;
const memcpy=@import_module("<memcpy/get_params>",.{.width=8,.height=8});
layout {@set_rectangle(8,8);for(@range(i16,8)) |y| {for(@range(i16,8)) |x| {@set_tile_code(x,y,"pe.csl",.{.memcpy_params=memcpy.get_params(x),.groups=groups});}}
@export_name("A",[*]f16,true);@export_name("B",[*]f16,true);@export_name("C",[*]f16,true);@export_name("progress",[*]u16,true);@export_name("queues",[*]u16,true);@export_name("init_task",fn()void);@export_name("hls_main",fn()void);}
''')
    bs=[];expected=[]
    for e in range(8):
        base=np.array([2048,0,1,-2048 if e%2==0 else 1.],float)*2**(-(e//2))
        y=np.zeros(8)
        if g==2:y[:4]=base
        else:y[1::2]=base
        v=y[:,None,None]*np.array([1,.5,-1,-.5,2,.25,-2,-.25])[None,:,None]
        a=np.concatenate([v,-v],axis=2);b=np.concatenate([a,a*.5,a*2,a*.25],axis=2);c=a[::-1].copy()
        bs.append({k:q.tolist() for k,q in [('A',a),('B',b),('C',c)]})
        expected.append({k:np.broadcast_to(grouped(q,8//g,(8//g)//2)[1],q.shape).astype(np.float16).view(np.uint16).tolist() for k,q in [('A',a),('B',b),('C',c)]})
    data={'inputs.json':bs,'expected.json':expected,'schema.json':dict(rows=8,cols=8,inputs=dict(A=2,B=8,C=2),outputs=dict(A=2,B=8,C=2,progress=1,queues=2),progress='progress',initialize='init_task',launch='hls_main'),'runtime-options.json':dict(suppress_trace=True,num_threads=8,dump_core=False),'sdk-command.json':['cslc','layout.csl','--arch=wse3','--fabric-dims=15,10','--fabric-offsets=4,1',f'--params=groups:{g}','-o=out','--memcpy','--channels=1']}
    for n,v in data.items():(root/n).write_text(json.dumps(v)+'\n')
    for src,name in [(Path(__file__),'driver.py'),(ROOT/'experiments/probe_runtime.py','probe_runtime.py'),(ROOT/'lib/Runtime/sdk_process.py','sdk_process.py')]:shutil.copyfile(src,root/name)
    (root/'provenance.json').write_text(json.dumps(dict(groups=g,source='Decode runtime Y routes and dynamic DSD lowering',files={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}),indent=2)+'\n');print(root.relative_to(ROOT))

if __name__=='__main__':
    if sys.argv[1]=='--prepare':prepare(int(sys.argv[2]))
    elif sys.argv[1]=='--worker':mesh_half_worker(Path(sys.argv[2]).resolve())
    else:
        root=Path(sys.argv[2]).resolve();execute(root,300)
        result=read(root/'results.json');expected=read(root/'expected.json');assert len(result['cases'])==len(expected)==8
        for r,v in zip(result['cases'],expected):
            for k,a in v.items():np.testing.assert_array_equal(r[k],a)
            assert np.all((np.asarray(r['queues'])&248)==248)
        (root/'order-review.json').write_text(json.dumps(dict(passed=True,epochs=8,extent_sequence=[2,8,2],results_sha256=sha(root/'results.json'),provenance_sha256=sha(root/'provenance.json')))+'\n');print('EXACT ORDER PASS')
