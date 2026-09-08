#include "spatial.hpp"
// Supplied rotated query and one shared read-only cache. No append or mask.
void design() {
 auto x=spatial::input<5,256,spatial::f16>("x",1.0);
 auto query=spatial::input<5,256,spatial::f16>("query",1.0);
 auto key=spatial::input<512,256,spatial::f16>("key",1.0);
 auto value=spatial::input<512,256,spatial::f16>("value",1.0);
 auto wo=spatial::input<256,256,spatial::f16>("wo",0.125);
 // Logical transpose is a view implemented by physical input serialization.
 auto kt=spatial::transpose(key);
#pragma csl dataflow rows=8 cols=8 broadcast=resident_columns axis=x reduce=sdk_axis result=sequence_rows replicas=columns fusion=none accumulation=f16 collective=f32 compute=dsr fp=relaxed
 auto score=spatial::matmul_blocked<spatial::f16,spatial::scalar>(query,kt,32);
#pragma csl dataflow rows=8 cols=8 partition=sequence axis=y layout=batch_major reduce=max_sum provider=sdk_axis accumulation=f16 collective=f32 math=sdk_half compute=dsr fp=relaxed
 auto probability=spatial::softmax(score,0.0625);
#pragma csl dataflow rows=8 cols=8 broadcast=resident_rows axis=y reduce=sdk_axis result=feature_columns replicas=rows fusion=none accumulation=f16 collective=f32 compute=dsr fp=relaxed
 auto context=spatial::matmul_blocked<spatial::f16,spatial::scalar>(probability,value,32);
#pragma csl dataflow rows=8 cols=8 broadcast=resident_columns axis=x reduce=sdk_axis result=feature_rows replicas=columns fusion=none accumulation=f16 collective=f32 compute=dsr fp=relaxed
 auto delta=spatial::matmul_blocked<spatial::f16,spatial::scalar>(context,wo,32);
#pragma csl dataflow rows=8 cols=8 layout=batch_major axis=y compute=dsr fp=relaxed
 auto result=spatial::add(x,delta);
 spatial::output("result",result);
}
