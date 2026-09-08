#include "spatial.hpp"
// One resident graph: supplied shared old cache; new K/V are outputs, not appended.
void design() {
 auto x=spatial::input<3,256,spatial::f16>("x",1.0);
 auto gamma=spatial::input<1,256,spatial::f16>("gamma",1.0);
 auto wq=spatial::input<256,256,spatial::f16>("wq",0.03125);
 auto wk=spatial::input<256,256,spatial::f16>("wk",0.03125);
 auto wv=spatial::input<256,256,spatial::f16>("wv",0.03125);
 auto cosine=spatial::input<1,128,spatial::f16>("cosine",1.0);
 auto sine=spatial::input<1,128,spatial::f16>("sine",1.0);
 auto key=spatial::input<512,256,spatial::f16>("key",1.0);
 auto value=spatial::input<512,256,spatial::f16>("value",1.0);
 auto wo=spatial::input<256,256,spatial::f16>("wo",0.125);
 auto wu=spatial::input<256,512,spatial::f16>("wu",0.03125);
 auto wg=spatial::input<256,512,spatial::f16>("wg",0.03125);
 auto wd=spatial::input<512,256,spatial::f16>("wd",0.00390625);
#pragma csl dataflow rows=16 cols=16 partition=features axis=y layout=batch_major reduce=sdk_axis result=replicated_columns accumulation=f16 collective=f32 math=sdk_half compute=dsr fp=relaxed
 auto normalized=spatial::rmsnorm(x,gamma,0.000001);
#pragma csl dataflow rows=16 cols=16 broadcast=resident_rows axis=y reduce=sdk_axis result=feature_columns replicas=rows fusion=collective accumulation=f16 collective=f32 compute=dsr fp=relaxed
 auto query=spatial::matmul_blocked<spatial::f16,spatial::scalar>(normalized,wq,1);
#pragma csl dataflow rows=16 cols=16 broadcast=resident_rows axis=y reduce=sdk_axis result=feature_columns replicas=rows fusion=collective accumulation=f16 collective=f32 compute=dsr fp=relaxed
 auto new_key=spatial::matmul_blocked<spatial::f16,spatial::scalar>(normalized,wk,16);
#pragma csl dataflow rows=16 cols=16 broadcast=resident_rows axis=y reduce=sdk_axis result=feature_columns replicas=rows fusion=collective accumulation=f16 collective=f32 compute=dsr fp=relaxed
 auto new_value=spatial::matmul_blocked<spatial::f16,spatial::scalar>(normalized,wv,16);
#pragma csl dataflow rows=16 cols=16 partition=features axis=x layout=batch_major coefficients=feature_pairs compute=dsr fp=relaxed
 auto rotated_query=spatial::rotate_pairs<spatial::pair_order::odd_even>(query,cosine,sine);
#pragma csl dataflow rows=16 cols=16 partition=features axis=x layout=batch_major coefficients=feature_pairs compute=dsr fp=relaxed
 auto rotated_key=spatial::rotate_pairs<spatial::pair_order::odd_even>(new_key,cosine,sine);
 auto kt=spatial::transpose(key);
#pragma csl dataflow rows=16 cols=16 broadcast=resident_columns axis=x reduce=sdk_axis result=sequence_rows replicas=columns fusion=none accumulation=f16 collective=f32 compute=dsr fp=relaxed
 auto score=spatial::matmul_blocked<spatial::f16,spatial::scalar>(rotated_query,kt,16);
#pragma csl dataflow rows=16 cols=16 partition=sequence axis=y layout=batch_major reduce=max_sum provider=sdk_axis accumulation=f16 collective=f32 math=sdk_half compute=dsr fp=relaxed
 auto probability=spatial::softmax(score,0.0625);
#pragma csl dataflow rows=16 cols=16 broadcast=resident_rows axis=y reduce=sdk_axis result=feature_columns replicas=rows fusion=none accumulation=f16 collective=f32 compute=dsr fp=relaxed
 auto context=spatial::matmul_blocked<spatial::f16,spatial::scalar>(probability,value,32);
#pragma csl dataflow rows=16 cols=16 broadcast=resident_columns axis=x reduce=sdk_axis result=feature_rows replicas=columns fusion=none accumulation=f16 collective=f32 compute=dsr fp=relaxed
 auto delta=spatial::matmul_blocked<spatial::f16,spatial::scalar>(context,wo,16);
#pragma csl dataflow rows=16 cols=16 layout=batch_major axis=y compute=dsr fp=relaxed
 auto result=spatial::add(x,delta);
#pragma csl dataflow rows=16 cols=16 partition=features axis=y layout=batch_major reduce=sdk_axis result=replicated_columns accumulation=f16 collective=f32 statistic=mean math=sdk_half compute=dsr fp=relaxed
 auto ffn_normalized=spatial::rmsnorm(result,gamma,0.000001);
#pragma csl dataflow rows=16 cols=16 broadcast=resident_rows axis=y reduce=sdk_axis result=feature_columns replicas=rows fusion=collective accumulation=f16 collective=f32 compute=dsr fp=relaxed
 auto up=spatial::matmul_blocked<spatial::f16,spatial::scalar>(ffn_normalized,wu,4);
#pragma csl dataflow rows=16 cols=16 broadcast=resident_rows axis=y reduce=sdk_axis result=feature_columns replicas=rows fusion=collective accumulation=f16 collective=f32 compute=dsr fp=relaxed
 auto gate=spatial::matmul_blocked<spatial::f16,spatial::scalar>(ffn_normalized,wg,4);
#pragma csl dataflow rows=16 cols=16 layout=batch_major axis=x compute=map math=sdk_stable_half fp=relaxed
 auto activation=spatial::silu(gate);
#pragma csl dataflow rows=16 cols=16 layout=batch_major axis=x compute=dsr fp=relaxed
 auto hidden=spatial::multiply(up,activation);
#pragma csl dataflow rows=16 cols=16 broadcast=resident_columns axis=x reduce=sdk_axis result=feature_rows replicas=columns fusion=none accumulation=f16 collective=f32 compute=dsr fp=relaxed
 auto ffn_delta=spatial::matmul_blocked<spatial::f16,spatial::scalar>(hidden,wd,4);
#pragma csl dataflow rows=16 cols=16 layout=batch_major axis=y compute=dsr fp=relaxed
 auto final_result=spatial::add(result,ffn_delta);
 spatial::output("result",final_result);
 spatial::output("new_key",rotated_key);
 spatial::output("new_value",new_value);
}
