#include "spatial.hpp"
void design(){
 auto q_weight=spatial::input<64,64,spatial::f16>("q_weight",0.00390625);
 auto k_weight=spatial::input<64,64,spatial::f16>("k_weight",0.00390625);
 auto v_weight=spatial::input<64,64,spatial::f16>("v_weight",0.00390625);
 auto output_weight=spatial::input<64,64,spatial::f16>("output_weight",0.0078125);
 auto input_x=spatial::input<64,64,spatial::f16>("input_x",0.125);
 auto gamma=spatial::input<1,64,spatial::f16>("gamma",1.5);
 auto u=spatial::input<64,256,spatial::f16>("up_weight",0.0078125);
 auto g=spatial::input<64,256,spatial::f16>("gate_weight",0.0078125);
 auto d=spatial::input<256,64,spatial::f16>("down_weight",0.0078125);
 auto cosine=spatial::input<1,32,spatial::f16>("cosine",1.0);
 auto sine=spatial::input<1,32,spatial::f16>("sine",1.0);
 #pragma csl dataflow rows=8 cols=8 partition=tiles reduce=bidirectional_chain weights=feature_columns accumulation=f16 math=sdk_half compute=dsr fp=relaxed
 auto input_normalized=spatial::rmsnorm(input_x,gamma,0.000001);
 #pragma csl dataflow rows=8 cols=8 exchange=two_hop initial_align=forward reduce=local overlap=double_buffer compute=dsr fp=relaxed
 auto q_raw=spatial::matmul(input_normalized,q_weight);
 #pragma csl dataflow rows=8 cols=8 exchange=two_hop initial_align=forward reduce=local overlap=double_buffer compute=dsr fp=relaxed
 auto k_raw=spatial::matmul(input_normalized,k_weight);
 #pragma csl dataflow rows=8 cols=8 exchange=two_hop initial_align=forward reduce=local overlap=double_buffer compute=dsr fp=relaxed
 auto v=spatial::matmul(input_normalized,v_weight);
 #pragma csl dataflow rows=8 cols=8 partition=tiles coefficients=feature_pairs compute=dsd fp=relaxed
 auto q=spatial::rotate_pairs<spatial::pair_order::odd_even>(q_raw,cosine,sine);
 #pragma csl dataflow rows=8 cols=8 partition=tiles coefficients=feature_pairs compute=dsd fp=relaxed
 auto k=spatial::rotate_pairs<spatial::pair_order::odd_even>(k_raw,cosine,sine);
 auto kt=spatial::transpose(k);
 #pragma csl dataflow rows=8 cols=8 exchange=vertical_two_hop reduce=rotating_root order=east_first overlap=double_buffer compute=dsr fp=relaxed
 auto score=spatial::matmul(q,kt);
 #pragma csl dataflow rows=8 cols=8 partition=tiles reduce=max_sum accumulation=f16 math=sdk_half compute=dsr fp=relaxed elementwise=map
 auto probability=spatial::softmax(score,0.125);
 #pragma csl dataflow rows=8 cols=8 exchange=two_hop initial_align=both_axes reduce=local overlap=double_buffer compute=dsr fp=relaxed
 auto attention=spatial::matmul(probability,v);
 #pragma csl dataflow rows=8 cols=8 exchange=two_hop initial_align=forward reduce=local overlap=double_buffer compute=dsr fp=relaxed
 auto projection=spatial::matmul(attention,output_weight);
 #pragma csl dataflow rows=8 cols=8 partition=tiles compute=dsr fp=relaxed
 auto z=spatial::add(projection,input_x);
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
