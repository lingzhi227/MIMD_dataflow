#include "spatial.hpp"
void design() {
 auto x=spatial::input<64,128,spatial::f16>("x");
 auto w=spatial::input<1,128,spatial::f16>("w");
 #pragma csl dataflow rows=8 cols=8 partition=tiles reduce=bidirectional_chain weights=feature_columns accumulation=f16 math=sdk_half compute=dsr fp=relaxed
 auto normalized=spatial::rmsnorm(x,w,0.000001);
 auto weight0=spatial::input<128,128,spatial::f16>("weight0");
 #pragma csl dataflow rows=8 cols=8 exchange=two_hop initial_align=forward reduce=local overlap=double_buffer fp=relaxed compute=dsr
 auto projection0=spatial::matmul(normalized,weight0);
 spatial::output("projection0",projection0);
 auto weight1=spatial::input<128,128,spatial::f16>("weight1");
 #pragma csl dataflow rows=8 cols=8 exchange=two_hop initial_align=forward reduce=local overlap=double_buffer fp=relaxed compute=dsr
 auto projection1=spatial::matmul(normalized,weight1);
 spatial::output("projection1",projection1);
 auto weight2=spatial::input<128,128,spatial::f16>("weight2");
 #pragma csl dataflow rows=8 cols=8 exchange=two_hop initial_align=forward reduce=local overlap=double_buffer fp=relaxed compute=dsr
 auto projection2=spatial::matmul(normalized,weight2);
 spatial::output("projection2",projection2);
}
