"""Write reviewed application sources/catalog; never used by the compiler backend."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


from pathlib import Path
import json

ROOT = repository_root(__file__)
catalog = []


def add(project, name, origins, source, fixture, contract, partitions=1):
    d = ROOT / "benchmarks" / project / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "hls.cpp").write_text(
        '#include "spatial.hpp"\nvoid design() {\n' + source + "\n}\n"
    )
    item = dict(
        project=project,
        kernel=name,
        origins=origins,
        fixture=fixture,
        contract=contract,
        partitions=partitions,
        status="source_ready",
    )
    (d / "PORT.json").write_text(json.dumps(item, indent=2) + "\n")
    catalog.append(item)


def local(inputs, outshape, body):
    decl = "\n".join(
        f'  auto {n}=spatial::input<{r},{c}>("{n}");' for n, r, c in inputs
    )
    params = ", ".join(f"const spatial::tensor<{r},{c}>& {n}" for n, r, c in inputs)
    return (
        decl
        + "\n  auto result=spatial::kernel("
        + ",".join(n for n, _, _ in inputs)
        + ", []("
        + params
        + ") {\n"
        + body
        + '\n  });\n  spatial::output("result",result);'
    )


# Preserve the already hand-written right-looking Cholesky source.
p = ROOT / "benchmarks/linear_algebra/sdk_examples/cholesky/hls.cpp"
ch = p.read_text()
item = dict(
    project="sdk_examples",
    kernel="cholesky",
    origins=["benchmarks/cholesky/pe.csl"],
    fixture="cholesky",
    contract="Right-looking lower Cholesky, 4x4 SPD; one local tile. Original multi-tile schedule not claimed.",
    partitions=1,
    status="source_ready",
)
catalog.append(item)
(p.parent / "PORT.json").write_text(json.dumps(item, indent=2) + "\n")
lu = """spatial::tensor<4,4> lu=a;
for(int k=0;k<4;++k){
 spatial::require(spatial::abs(lu.data[k*4+k])>0.00001f);
 for(int i=k+1;i<4;++i){
  lu.data[i*4+k]/=lu.data[k*4+k];
  for(int j=k+1;j<4;++j){lu.data[i*4+j]-=lu.data[i*4+k]*lu.data[k*4+j];}
 }
}
return lu;"""
add(
    "matrix_algorithms",
    "lu_no_pivot",
    [
        "lu_factorization/one_element_per_pe/pe_program.csl",
        "lu_factorization/many_elements_per_pe/pe_program.csl",
    ],
    local([("a", 4, 4)], (4, 4), lu),
    "lu",
    "No-pivot elimination with packed L/U; 4x4. Two original layouts share equation coverage, not route equivalence.",
)
qr = """spatial::tensor<8,4> out{};
for(int i=0;i<4;++i){for(int j=0;j<4;++j){out.data[i*4+j]=a.data[i*4+j];if(i==j){out.data[16+i*4+j]=1.0f;}}}
for(int k=0;k<4;++k){for(int i=k+1;i<4;++i){
 float av=out.data[k*4+k];float bv=out.data[i*4+k];float cs=1.0f;float sn=0.0f;
 if(bv!=0.0f){if(spatial::abs(bv)>spatial::abs(av)){float tau=-av/bv;sn=1.0f/spatial::sqrt(1.0f+tau*tau);cs=sn*tau;}else{float tau=-bv/av;cs=1.0f/spatial::sqrt(1.0f+tau*tau);sn=cs*tau;}}
 for(int j=0;j<4;++j){float top=out.data[k*4+j];float bottom=out.data[i*4+j];out.data[k*4+j]=cs*top-sn*bottom;out.data[i*4+j]=sn*top+cs*bottom;
 float qt=out.data[16+k*4+j];float qb=out.data[16+i*4+j];out.data[16+k*4+j]=cs*qt-sn*qb;out.data[16+i*4+j]=sn*qt+cs*qb;}
}}
return out;"""
add(
    "matrix_algorithms",
    "qr_givens",
    [
        "QR_factorization/many_elements_per_pe/pe_program.csl",
        "QR_factorization/one_element_per_pe/pe_program.csl",
    ],
    local([("a", 4, 4)], (8, 4), qr),
    "qr",
    "Stable source Givens coefficient branches; return R and accumulated Q-transpose. Explicit pivot-row rotation order; original distributed ordering/layout not preserved.",
)
matmul = """auto a=spatial::input<4,4>("a");auto b=spatial::input<4,4>("b");auto result=spatial::matmul(a,b);spatial::output("result",result);"""
for project, name, origins in [
    ("sdk_examples", "gemm", ["benchmarks/gemm-collectives_2d/pe.csl"]),
    ("waferllm", "summa_equation", ["SUMMA/src/summa.csl"]),
    ("waferllm", "meshgemm_equation", ["MeshGEMM/src/meshgemm.csl"]),
    ("matrix_algorithms", "cannon_equation", ["Cannons_algorithm/pe_program.csl"]),
]:
    add(
        project,
        name,
        origins,
        matmul,
        "gemm",
        "Matrix product equation, f32 4x4. Split-K is our schedule; named upstream Cannon/MeshGEMM schedules remain unported.",
        2,
    )
for project, name, origins in [
    (
        "sdk_examples",
        "gemv",
        [
            "benchmarks/gemv-collectives_2d/pe.csl",
            "benchmarks/gemv-checkerboard-pattern/pe.csl",
            "benchmarks/single-tile-matvec/src/pe_matvec.csl",
        ],
    ),
    ("waferllm", "meshgemv_equation", ["MeshGEMV/src/meshgemv.csl"]),
    (
        "spada",
        "gemv",
        ["samples/spatial/blas/gemv.sptl", "samples/spatial/blas/matvec.sptl"],
    ),
]:
    add(
        project,
        name,
        origins,
        'auto a=spatial::input<4,4>("a");auto b=spatial::input<4,1>("b");auto result=spatial::matmul(a,b);spatial::output("result",result);',
        "gemv",
        "Matrix-vector equation; f32 instead of WaferLLM original half profile where applicable.",
        2,
    )
add(
    "spada",
    "axpy",
    ["samples/spatial/blas/axpy.sptl"],
    'auto x=spatial::input<1,4>("x");auto y=spatial::input<1,4>("y");auto ax=spatial::map(x,[](float x){return 2.5f*x;});auto result=spatial::add(ax,y);spatial::output("result",result);',
    "axpy",
    "Alpha specialized to 2.5; four-element local vectors, graph map/add actors.",
)
res = """auto a=spatial::input<4,4>("a");auto x=spatial::input<4,1>("x");auto b=spatial::input<4,1>("b");auto ax=spatial::matmul(a,x);auto negative=spatial::map(ax,[](float x){return -x;});auto result=spatial::add(b,negative);spatial::output("result",result);"""
add(
    "sdk_examples",
    "residual",
    ["benchmarks/residual/residual.csl"],
    res,
    "residual",
    "b-Ax residual equation; dense four-row counterpart, not original structured operator storage.",
)
# Explicit collective topologies: actual HLS graph edges define the sum order.
base = "\n".join(f'auto p{i}=spatial::input<1,4>("p{i}");' for i in range(4))
chain = (
    base
    + '\nauto s2=spatial::add(p2,p3);auto s1=spatial::add(p1,s2);auto result=spatial::add(p0,s1);spatial::output("result",result);'
)
tree = (
    base
    + '\nauto left=spatial::add(p0,p1);auto right=spatial::add(p2,p3);auto result=spatial::add(left,right);spatial::output("result",result);'
)
for project, paths in [
    ("spatial_collectives", ["modules/chain.csl", "modules/tree_runtime.csl"]),
    (
        "spada",
        [
            "samples/spatial/collectives/chain_reduce_1D.sptl",
            "samples/spatial/collectives/tree_reduce_1D.sptl",
        ],
    ),
]:
    add(
        project,
        "chain_reduce",
        [paths[0]],
        chain,
        "collective",
        "Four-participant right-to-left ordered vector sum; intermediate actors separate transport from numerical stage.",
    )
    add(
        project,
        "tree_reduce",
        [paths[1]],
        tree,
        "collective",
        "Four-participant balanced binary vector sum; no performance-equivalence claim.",
    )
broadcast = 'auto x=spatial::input<1,4>("x");' + "".join(
    f'spatial::output("p{i}",x);' for i in range(4)
)
for project, origin in [
    ("spatial_collectives", "modules/broadcast.csl"),
    ("spada", "samples/spatial/collectives/broadcast_1D.sptl"),
    ("sdk_examples", "benchmarks/row-col-broadcast/src/kernel.csl"),
]:
    add(
        project,
        "broadcast",
        [origin],
        broadcast,
        "broadcast",
        "Four receiver ports; compiler fanout tree and owned-transfer runtime. Multicast routing policy not identical to upstream.",
    )
allreduce = tree.replace(
    'spatial::output("result",result);',
    "".join(f'spatial::output("p{i}",result);' for i in range(4)),
)
add(
    "spada",
    "allreduce",
    ["samples/spatial/collectives/allreduce_1D.sptl"],
    allreduce,
    "allreduce",
    "Balanced reduction followed by graph broadcast, four participants.",
)
# WaferLLM stage-level kernels; explicitly separate mathematical/f32 from original half arithmetic.
rms = """spatial::tensor<2,4> out{};for(int i=0;i<2;++i){float sum=0.0f;for(int j=0;j<4;++j){sum+=x.data[i*4+j]*x.data[i*4+j];}float inv=1.0f/spatial::sqrt(sum/4.0f+0.000001f);for(int j=0;j<4;++j){out.data[i*4+j]=x.data[i*4+j]*inv*w.data[j];}}return out;"""
add(
    "waferllm",
    "rmsnorm",
    ["Prefill/src/prefill.csl", "Decode/src/decode.csl"],
    local([("x", 2, 4), ("w", 1, 4)], (2, 4), rms),
    "rmsnorm",
    "Per-row RMS normalization with weight feature axis and epsilon 1e-6; f32 mathematical profile.",
)
softmax = """spatial::tensor<2,4> out{};for(int i=0;i<2;++i){float largest=x.data[i*4];for(int j=1;j<4;++j){if(x.data[i*4+j]>largest){largest=x.data[i*4+j];}}float sum=0.0f;for(int j=0;j<4;++j){out.data[i*4+j]=spatial::exp(x.data[i*4+j]-largest);sum+=out.data[i*4+j];}for(int j=0;j<4;++j){out.data[i*4+j]/=sum;}}return out;"""
add(
    "waferllm",
    "softmax",
    ["Prefill/src/prefill.csl", "Decode/src/decode.csl"],
    local([("x", 2, 4)], (2, 4), softmax),
    "softmax",
    "Stable row softmax; no mask or model inferred; f32 math exp implementation.",
)
silu = """spatial::tensor<2,4> out{};for(int i=0;i<8;++i){out.data[i]=up.data[i]*gate.data[i]/(1.0f+spatial::exp(-gate.data[i]));}return out;"""
add(
    "waferllm",
    "gated_silu",
    ["Prefill/src/prefill.csl", "Decode/src/decode.csl"],
    local([("up", 2, 4), ("gate", 2, 4)], (2, 4), silu),
    "silu",
    "Up times SiLU(gate), f32 stage profile; weights/projections supplied separately.",
)
rope = """spatial::tensor<2,4> out{};for(int i=0;i<2;++i){for(int pair=0;pair<2;++pair){float even=x.data[i*4+pair*2];float odd=x.data[i*4+pair*2+1];float cs=coeff.data[pair*2];float sn=coeff.data[pair*2+1];out.data[i*4+pair*2]=even*cs-odd*sn;out.data[i*4+pair*2+1]=odd*cs+even*sn;}}return out;"""
add(
    "waferllm",
    "rope",
    ["Prefill/src/prefill.csl", "Decode/src/decode.csl"],
    local([("x", 2, 4), ("coeff", 1, 4)], (2, 4), rope),
    "rope",
    "Supplied rotary coefficients; intended even/odd mathematics, not erroneous original scratch/index behavior.",
)
(ROOT / "benchmarks/catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
