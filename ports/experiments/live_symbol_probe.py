"""Test SDK debug observation boundaries against normal memcpy on a qualified RMS ELF."""
import sys,json,os,shutil,datetime,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def prepare():
    bundle=ROOT/'projects/waferllm/batched_rms_3x512_8x8_g2/run-20260907T160529124015Z'
    root=ROOT/'evidence'/('live-symbol-probe-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));root.mkdir()
    shutil.copytree(bundle/'out',root/'out');shutil.copytree(bundle/'implementation',root/'implementation')
    for name in ('schedule.json','semantic.json','batches.json'):shutil.copyfile(bundle/name,root/name)
    for name in ('probe_runtime.py',):shutil.copyfile(ROOT/'experiments'/name,root/name)
    shutil.copyfile(ROOT/'toolchain/sdk_process.py',root/'sdk_process.py');shutil.copyfile(__file__,root/'driver.py')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (root/'provenance.json').write_text(json.dumps(dict(source_bundle=str(bundle.relative_to(ROOT)),files={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}),indent=2)+'\n');print(root.relative_to(ROOT))

def worker(root):
    os.chdir(root);sys.path.insert(0,str(root/'implementation'))
    import numpy as np
    from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime,SimfabConfig,SdkTarget,get_platform,MemcpyDataType,MemcpyOrder
    from cerebras.sdk.sdk_utils import input_array_to_u32
    from mesh_batched_rms_sdk import packed
    read=lambda n:json.loads((root/n).read_text())
    s,m,bs=[read(n) for n in ('schedule.json','semantic.json','batches.json')]
    runner=SdkRuntime('out',get_platform(None,SimfabConfig(num_threads=8,suppress_trace=True,dump_core=False),SdkTarget.WSE3))
    runner.load();runner.run();runner.launch('init_task',nonblock=False)
    records=[]
    def attempt(stage,call):
        try:
            value=call();record=dict(stage=stage,success=True,value=None if value is None else np.asarray(value).tolist())
        except Exception as e:record=dict(stage=stage,success=False,error=repr(e))
        records.append(record);(root/'results.json').write_text(json.dumps(dict(success=False,checks=records))+'\n');print(stage,record['success'],flush=True)
    try:
        for e in range(2):
            data=packed(s,m,bs[e])
            for name,a in data.items():runner.memcpy_h2d(runner.get_id(name),input_array_to_u32(np.asarray(a,np.float16).ravel(),1,1),0,0,8,8,a.shape[-1],streaming=False,data_type=MemcpyDataType.MEMCPY_16BIT,order=MemcpyOrder.ROW_MAJOR,nonblock=False)
            runner.launch('hls_main',nonblock=False)
            attempt(f'live_read_{e}',lambda:runner.read_symbol(4,1,'X',dtype='uint16'))
            attempt(f'dump_core_{e}',lambda:runner.dump_core(str(root/f'snapshot{e}.core')))
            attempt(f'after_dump_read_{e}',lambda:runner.read_symbol(4,1,'X',dtype='uint16'))
            attempt(f'live_elf_dump_{e}',lambda:runner.dump_elf_core(str(root/f'elfsnapshot{e}')))
            a=np.zeros(192,np.uint32);runner.memcpy_d2h(a,runner.get_id('X'),0,0,1,1,192,streaming=False,data_type=MemcpyDataType.MEMCPY_16BIT,order=MemcpyOrder.ROW_MAJOR,nonblock=False)
            expected=np.asarray(data['X'][0,0],np.float16).view(np.uint16);np.testing.assert_array_equal(a,expected)
            for row in records:
                if row['stage'] in (f'live_read_{e}',f'after_dump_read_{e}') and row['success']:
                    np.testing.assert_array_equal(np.asarray(row['value']).ravel(),a)
            records.append(dict(stage=f'memcpy_reference_{e}',success=True,value=a.tolist()))
    finally:runner.stop()
    attempt('stopped_read',lambda:runner.read_symbol(4,1,'X',dtype='uint16'))
    (root/'results.json').write_text(json.dumps(dict(success=True,checks=records))+'\n')

if __name__=='__main__':
    if sys.argv[1]=='--prepare':prepare()
    elif sys.argv[1]=='--worker':worker(Path(sys.argv[2]).resolve())
    else:
        from probe_runtime import execute
        execute(Path(sys.argv[2]).resolve(),300)
