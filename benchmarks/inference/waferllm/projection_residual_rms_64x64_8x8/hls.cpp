#include "spatial.hpp"
void design(){
 auto x=spatial::input<64,64,spatial::f16>("activation",0.125);
 auto w=spatial::input<64,64,spatial::f16>("weight",0.125);
 auto r=spatial::input<64,64,spatial::f16>("residual",0.5);
 auto g=spatial::input<1,64,spatial::f16>("gamma",1.5);
 #pragma csl dataflow rows=8 cols=8 exchange=two_hop initial_align=forward reduce=local overlap=double_buffer compute=dsr fp=relaxed
 auto projected=spatial::matmul(x,w);
 #pragma csl dataflow rows=8 cols=8 partition=tiles compute=dsr fp=relaxed
 auto z=spatial::add(projected,r);
 #pragma csl dataflow rows=8 cols=8 partition=tiles reduce=bidirectional_chain weights=feature_columns accumulation=f16 math=sdk_half compute=dsr fp=relaxed
 auto normalized=spatial::rmsnorm(z,g,0.000001);
 spatial::output("output",normalized);
}
