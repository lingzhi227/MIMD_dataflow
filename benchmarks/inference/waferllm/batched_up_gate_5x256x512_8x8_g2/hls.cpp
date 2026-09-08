#include "spatial.hpp"
void design() {
 auto x=spatial::input<5,256,spatial::f16>("x",1.0);
 auto w=spatial::input<1,256,spatial::f16>("w",1.0);
 auto weight0=spatial::input<256,512,spatial::f16>("weight0",0.125);
 auto weight1=spatial::input<256,512,spatial::f16>("weight1",0.125);
#pragma csl dataflow rows=8 cols=8 partition=features axis=y layout=batch_major reduce=grouped_two_tree groups=2 result=replicated_columns accumulation=f16 math=sdk_half compute=dsr fp=relaxed
 auto normalized=spatial::rmsnorm(x,w,0.000001);
#pragma csl dataflow rows=8 cols=8 broadcast=resident_rows reduce=grouped_two_tree groups=2 result=feature_columns replicas=rows fusion=collective compute=dsr fp=relaxed
 auto branch0=spatial::matmul(normalized,weight0);
 spatial::output("branch0",branch0);
#pragma csl dataflow rows=8 cols=8 broadcast=resident_rows reduce=grouped_two_tree groups=2 result=feature_columns replicas=rows fusion=collective compute=dsr fp=relaxed
 auto branch1=spatial::matmul(normalized,weight1);
 spatial::output("branch1",branch1);
}
