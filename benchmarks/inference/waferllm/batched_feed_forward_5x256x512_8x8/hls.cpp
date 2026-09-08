#include "spatial.hpp"
// Decode-style resident FFN; explicit half local arithmetic, SDK f32 collectives.
void design() {
 auto x=spatial::input<5,256,spatial::f16>("x",1.0);
 auto gamma=spatial::input<1,256,spatial::f16>("gamma",1.0);
 auto wu=spatial::input<256,512,spatial::f16>("wu",0.03125);
 auto wg=spatial::input<256,512,spatial::f16>("wg",0.03125);
 auto wd=spatial::input<512,256,spatial::f16>("wd",0.00390625);
#pragma csl dataflow rows=8 cols=8 partition=features axis=y layout=batch_major reduce=sdk_axis result=replicated_columns accumulation=f16 collective=f32 math=sdk_half compute=dsr fp=relaxed
 auto norm=spatial::rmsnorm(x,gamma,0.000001);
#pragma csl dataflow rows=8 cols=8 broadcast=resident_rows axis=y reduce=sdk_axis result=feature_columns replicas=rows fusion=collective accumulation=f16 collective=f32 compute=dsr fp=relaxed
 auto up=spatial::matmul(norm,wu);
#pragma csl dataflow rows=8 cols=8 broadcast=resident_rows axis=y reduce=sdk_axis result=feature_columns replicas=rows fusion=collective accumulation=f16 collective=f32 compute=dsr fp=relaxed
 auto gate=spatial::matmul(norm,wg);
#pragma csl dataflow rows=8 cols=8 layout=batch_major axis=x compute=map math=sdk_stable_half fp=relaxed
 auto activation=spatial::silu(gate);
#pragma csl dataflow rows=8 cols=8 layout=batch_major axis=x compute=dsr fp=relaxed
 auto hidden=spatial::multiply(up,activation);
#pragma csl dataflow rows=8 cols=8 broadcast=resident_columns axis=x reduce=sdk_axis result=feature_rows replicas=columns fusion=none accumulation=f16 collective=f32 compute=dsr fp=relaxed
 auto delta=spatial::matmul(hidden,wd);
#pragma csl dataflow rows=8 cols=8 layout=batch_major axis=y compute=dsr fp=relaxed
 auto result=spatial::add(x,delta);
 spatial::output("result",result);
}
