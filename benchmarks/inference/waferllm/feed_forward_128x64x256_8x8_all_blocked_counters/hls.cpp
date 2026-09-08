#include "spatial.hpp"
void design(){
 auto z=spatial::input<128,64,spatial::f16>("z",0.125);
 auto gamma=spatial::input<1,64,spatial::f16>("gamma",1.5);
 auto u=spatial::input<64,256,spatial::f16>("up_weight",0.0078125);
 auto g=spatial::input<64,256,spatial::f16>("gate_weight",0.0078125);
 auto d=spatial::input<256,64,spatial::f16>("down_weight",0.0078125);
 #pragma csl dataflow rows=8 cols=8 partition=tiles reduce=bidirectional_chain weights=feature_columns accumulation=f16 math=sdk_half compute=dsr fp=relaxed
 auto x=spatial::rmsnorm(z,gamma,0.000001);
 #pragma csl dataflow rows=8 cols=8 exchange=two_hop initial_align=forward reduce=local overlap=double_buffer compute=dsr fp=relaxed accumulation=block_f32
 auto up=spatial::matmul_blocked<spatial::f16,spatial::scalar>(x,u,8);
 #pragma csl dataflow rows=8 cols=8 exchange=two_hop initial_align=forward reduce=local overlap=double_buffer compute=dsr fp=relaxed accumulation=block_f32
 auto gate=spatial::matmul_blocked<spatial::f16,spatial::scalar>(x,g,8);
 #pragma csl dataflow rows=8 cols=8 partition=tiles math=sdk_half compute=dsr elementwise=map fp=relaxed
 auto act=spatial::silu(gate);
 #pragma csl dataflow rows=8 cols=8 partition=tiles math=sdk_half compute=dsr elementwise=map fp=relaxed
 auto hidden=spatial::multiply(up,act);
 #pragma csl dataflow rows=8 cols=8 exchange=two_hop initial_align=forward reduce=local overlap=double_buffer compute=dsr fp=relaxed accumulation=block_f32
 auto delta=spatial::matmul_blocked<spatial::f16,spatial::scalar>(hidden,d,32);
 #pragma csl dataflow rows=8 cols=8 partition=tiles compute=dsr fp=relaxed
 auto result=spatial::add(z,delta);
 spatial::output("output",result);
}
