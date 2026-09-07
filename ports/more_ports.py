from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
catalog = json.loads((ROOT / "catalog.json").read_text())


def add(project, name, origins, inputs, body, fixture, contract):
    source = '#include "spatial.hpp"\nvoid design(){\n' + "\n".join(
        f'auto {n}=spatial::input<{r},{c}>("{n}");' for n, r, c in inputs
    )
    source += (
        "\nauto result=spatial::kernel("
        + ",".join(n for n, _, _ in inputs)
        + ", []("
        + ",".join(f"const spatial::tensor<{r},{c}>& {n}" for n, r, c in inputs)
        + "){\n"
        + body
        + '\n});spatial::output("result",result);}\n'
    )
    d = ROOT / "projects" / project / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "hls.cpp").write_text(source)
    item = dict(
        project=project,
        kernel=name,
        origins=origins,
        fixture=fixture,
        contract=contract,
        partitions=1,
        status="source_ready",
    )
    (d / "PORT.json").write_text(json.dumps(item, indent=2) + "\n")
    catalog[:] = [x for x in catalog if (x["project"], x["kernel"]) != (project, name)]
    catalog.append(item)


mc = """spatial::tensor<4,6> out=particles;
for(int p=0;p<4;++p){float energy=particles.data[p*6];for(int n=0;n<2;++n){
 spatial::require(energy>=table.data[n*4] && energy<=table.data[n*4+3]);
 int lower=n*4;
 for(int j=n*4;j<n*4+3;++j){if(energy>=table.data[j]){lower=j;}}
 float lo=table.data[lower];float hi=table.data[lower+1];spatial::require(hi>lo);
 float f=(hi-energy)/(hi-lo);
 for(int xs=0;xs<5;++xs){float low=table.data[8+lower*5+xs];float high=table.data[8+(lower+1)*5+xs];out.data[p*6+1+xs]+=table.data[48+n]*(high-f*(high-low));}
}}return out;"""
add(
    "mc_transport",
    "linear_xs_lookup",
    ["csl/full_implementation/device_code.csl"],
    [("particles", 4, 6), ("table", 1, 50)],
    mc,
    "mc_xs",
    "calculate_xs deterministic scalar-interpolation branch, two nuclides/four gridpoints/five XS components. Search specialized to bounded scan; particle sorting, diffusion and stochastic interpolation remain pending.",
)
flip = """spatial::tensor<1,4> out{};for(int i=0;i<4;++i){spatial::require(spins.data[i]==0.0f || spins.data[i]==1.0f);float e=q.data[i*4+i];for(int j=0;j<4;++j){if(i!=j){e+=q.data[i*4+j]*spins.data[j];}}if(spins.data[i]>0.0f){e=-e;}out.data[i]=e;}return out;"""
add(
    "annealing",
    "flip_energy",
    ["src/pe_program.csl"],
    [("q", 4, 4), ("spins", 1, 4)],
    flip,
    "flip",
    "prepare_flip_energy plus sign correction; four independent candidate indices on unchanged binary state. Q uses source combined off-diagonal convention; packed-bit storage and RNG not ported.",
)
accept = """spatial::tensor<4,1> out{};for(int i=0;i<4;++i){float d=values.data[i*3];float t=values.data[i*3+1];float draw=values.data[i*3+2];spatial::require(t>0.0f && draw>=0.0f && draw<1.0f);if(d<0.0f || draw<spatial::exp(-d/t)){out.data[i]=1.0f;}}return out;"""
add(
    "annealing",
    "acceptance",
    ["src/pe_program.csl"],
    [("values", 4, 3)],
    accept,
    "accept",
    "prepare_do_flip conditional acceptance with explicit supplied uniform draws; not a replacement PRNG or full annealer.",
)
# CG: bounded full solve on a four-row SPD operator. Device loop and convergence are explicit.
cg = """spatial::tensor<4,1> x{};spatial::tensor<4,1> r=b;spatial::tensor<4,1> p=b;spatial::tensor<4,1> w{};
float rr=0.0f;for(int i=0;i<4;++i){rr+=r.data[i]*r.data[i];}
for(int iter=0;iter<8;++iter){if(rr>0.000000000001f){
 for(int i=0;i<4;++i){w.data[i]=0.0f;for(int j=0;j<4;++j){w.data[i]+=a.data[i*4+j]*p.data[j];}}
 float pw=0.0f;for(int i=0;i<4;++i){pw+=p.data[i]*w.data[i];}spatial::require(pw>0.0f);float alpha=rr/pw;
 for(int i=0;i<4;++i){x.data[i]+=alpha*p.data[i];r.data[i]-=alpha*w.data[i];}
 float next=0.0f;for(int i=0;i<4;++i){next+=r.data[i]*r.data[i];}float beta=next/rr;for(int i=0;i<4;++i){p.data[i]=r.data[i]+beta*p.data[i];}rr=next;
}}return x;"""
add(
    "sdk_examples",
    "conjugate_gradient",
    ["benchmarks/conjugate-gradient/src/kernel_cg.csl"],
    [("a", 4, 4), ("b", 4, 1)],
    cg,
    "cg",
    "CG recurrence on dense representation of a small SPD operator, maximum eight iterations, squared residual threshold 1e-12, zero initial guess. Original distributed 7-point storage and global-reduction schedule remain pending.",
)
(ROOT / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
