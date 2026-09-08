
from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)

from pathlib import Path
import json

ROOT = repository_root(__file__)
catalog = json.loads((ROOT / "benchmarks/catalog.json").read_text())


def add(name, origin, body, fixture, contract):
    d = ROOT / "benchmarks/sdk_examples" / name
    d.mkdir(exist_ok=True)
    s = (
        '#include "spatial.hpp"\nvoid design(){auto a=spatial::input<4,4>("a");auto b=spatial::input<4,1>("b");auto result=spatial::kernel(a,b,[](const spatial::tensor<4,4>&a,const spatial::tensor<4,1>&b){\n'
        + body
        + '\n});spatial::output("result",result);}\n'
    )
    (d / "hls.cpp").write_text(s)
    item = dict(
        project="sdk_examples",
        kernel=name,
        origins=[origin],
        fixture=fixture,
        contract=contract,
        partitions=1,
        status="source_ready",
    )
    (d / "PORT.json").write_text(json.dumps(item, indent=2) + "\n")
    catalog[:] = [
        x for x in catalog if (x["project"], x["kernel"]) != ("sdk_examples", name)
    ]
    catalog.append(item)


pcg = """spatial::tensor<4,1> x{};spatial::tensor<4,1> r=b;spatial::tensor<4,1> z{};spatial::tensor<4,1> p{};spatial::tensor<4,1> w{};
float rho=0.0f;for(int i=0;i<4;++i){spatial::require(a.data[i*4+i]>0.0f);z.data[i]=r.data[i]/a.data[i*4+i];p.data[i]=z.data[i];rho+=r.data[i]*z.data[i];}
for(int iter=0;iter<8;++iter){float rr=0.0f;for(int i=0;i<4;++i){rr+=r.data[i]*r.data[i];}if(rr>0.000000000001f){
 for(int i=0;i<4;++i){w.data[i]=0.0f;for(int j=0;j<4;++j){w.data[i]+=a.data[i*4+j]*p.data[j];}}
 float pw=0.0f;for(int i=0;i<4;++i){pw+=p.data[i]*w.data[i];}spatial::require(pw>0.0f);float alpha=rho/pw;
 for(int i=0;i<4;++i){x.data[i]+=alpha*p.data[i];r.data[i]-=alpha*w.data[i];z.data[i]=r.data[i]/a.data[i*4+i];}
 float next=0.0f;for(int i=0;i<4;++i){next+=r.data[i]*z.data[i];}float beta=next/rho;for(int i=0;i<4;++i){p.data[i]=z.data[i]+beta*p.data[i];}rho=next;
}}return x;"""
add(
    "pcg_jacobi",
    "benchmarks/preconditioned-conjugate-gradient/src/kernel_pcg.csl",
    pcg,
    "cg",
    "Source Jacobi-PCG recurrence, zero initial x, at most eight iterations, small tridiagonal specialization of structured operator. Distributed SpMV/reduction scheduling pending.",
)
power = """spatial::tensor<4,1> x=b;spatial::tensor<4,1> y{};
for(int iter=0;iter<8;++iter){for(int i=0;i<4;++i){y.data[i]=0.0f;for(int j=0;j<4;++j){y.data[i]+=a.data[i*4+j]*x.data[j];}}
 float norm=0.0f;for(int i=0;i<4;++i){norm+=y.data[i]*y.data[i];}spatial::require(norm>0.0f);float inv=1.0f/spatial::sqrt(norm);for(int i=0;i<4;++i){x.data[i]=y.data[i]*inv;}}
return x;"""
add(
    "power_method",
    "benchmarks/power-method/src/kernel_power.csl",
    power,
    "power",
    "Eight original SpMV/norm/scale iterations, supplied initial vector; tridiagonal operator. No converged-eigenvector claim.",
)
bicg = """spatial::tensor<4,1> x{};spatial::tensor<4,1> r=b;spatial::tensor<4,1> p{};spatial::tensor<4,1> v{};spatial::tensor<4,1> s{};spatial::tensor<4,1> t{};
float old=1.0f;float alpha=1.0f;float omega=1.0f;
for(int iter=0;iter<16;++iter){float rr=0.0f;for(int i=0;i<4;++i){rr+=r.data[i]*r.data[i];}if(rr>0.000000000001f){
 float rho=0.0f;for(int i=0;i<4;++i){rho+=b.data[i]*r.data[i];}spatial::require(rho!=0.0f && omega!=0.0f);float beta=(rho/old)*(alpha/omega);
 for(int i=0;i<4;++i){p.data[i]=r.data[i]+beta*(p.data[i]-omega*v.data[i]);}
 for(int i=0;i<4;++i){v.data[i]=0.0f;for(int j=0;j<4;++j){v.data[i]+=a.data[i*4+j]*p.data[j];}}
 float bv=0.0f;for(int i=0;i<4;++i){bv+=b.data[i]*v.data[i];}spatial::require(bv!=0.0f);alpha=rho/bv;float ss=0.0f;
 for(int i=0;i<4;++i){s.data[i]=r.data[i]-alpha*v.data[i];ss+=s.data[i]*s.data[i];}
 if(ss<0.000000000001f){for(int i=0;i<4;++i){x.data[i]+=alpha*p.data[i];r.data[i]=s.data[i];}}
 else{for(int i=0;i<4;++i){t.data[i]=0.0f;for(int j=0;j<4;++j){t.data[i]+=a.data[i*4+j]*s.data[j];}}
 float ts=0.0f;float tt=0.0f;for(int i=0;i<4;++i){ts+=t.data[i]*s.data[i];tt+=t.data[i]*t.data[i];}spatial::require(tt>0.0f);omega=ts/tt;
 for(int i=0;i<4;++i){x.data[i]+=alpha*p.data[i]+omega*s.data[i];r.data[i]=s.data[i]-omega*t.data[i];}}
 old=rho;
}}return x;"""
add(
    "bicgstab",
    "benchmarks/bicgstab/src/kernel_bicgstab.csl",
    bicg,
    "bicgstab",
    "BiCGStab with explicit breakdown checks, zero initial x and at most sixteen iterations. Nonsymmetric tridiagonal specialization; original distributed communication remains pending.",
)
(ROOT / "benchmarks/catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
