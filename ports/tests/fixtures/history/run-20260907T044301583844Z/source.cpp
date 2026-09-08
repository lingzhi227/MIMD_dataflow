#include "spatial.hpp"
void design() {
 auto x=spatial::input<64,64,spatial::f16>("x",0.125);
 auto u=spatial::input<64,256,spatial::f16>("up_weight",0.125);
 auto g=spatial::input<64,256,spatial::f16>("gate_weight",0.125);
 auto d=spatial::input<256,64,spatial::f16>("down_weight",0.125);
 #pragma csl dataflow rows=8 cols=8 exchange=two_hop initial_align=forward reduce=local overlap=double_buffer compute=dsr fp=relaxed
 auto up=spatial::matmul(x,u);
 #pragma csl dataflow rows=8 cols=8 exchange=two_hop initial_align=forward reduce=local overlap=double_buffer compute=dsr fp=relaxed
 auto gate=spatial::matmul(x,g);
 #pragma csl dataflow rows=8 cols=8 partition=tiles math=sdk_half compute=dsr elementwise=map fp=relaxed
 auto activated=spatial::silu(gate);
 #pragma csl dataflow rows=8 cols=8 partition=tiles math=sdk_half compute=dsr elementwise=map fp=relaxed
 auto hidden=spatial::multiply(up,activated);
 #pragma csl dataflow rows=8 cols=8 exchange=two_hop initial_align=forward reduce=local overlap=double_buffer compute=dsr fp=relaxed
 auto result=spatial::matmul(hidden,d);
 spatial::output("output",result);
}
