#include "spatial.hpp"

namespace precision_experiment {
template<class O,int R,int K,int C,class A,class B>
auto product(const spatial::tensor<R,K,A>& a,const spatial::tensor<K,C,B>& b){
 spatial::tensor<R,C,O> out;
 for(int i=0;i<R;++i)for(int j=0;j<C;++j){
  float total=0;
  for(int k=0;k<K;++k)total=std::fma(float(a.data[i*K+k]),float(b.data[k*C+j]),total);
  out.data[i*C+j]=O(total);
 }return out;
}
template<class O,int R,int C,class A,class B>
auto add(const spatial::tensor<R,C,A>&a,const spatial::tensor<R,C,B>&b){
 spatial::tensor<R,C,O> out;
 for(int i=0;i<R*C;++i)out.data[i]=O(float(a.data[i])+float(b.data[i]));return out;
}
template<int R,int C,class A>
auto norm(const spatial::tensor<R,C,A>&x,const spatial::tensor<1,C,spatial::f16>&gamma,double epsilon){
 spatial::tensor<R,C,spatial::f16> out;
 for(int i=0;i<R;++i){double sum=0;for(int j=0;j<C;++j)sum+=double(x.data[i*C+j])*double(x.data[i*C+j]);
  double inv=1/std::sqrt(sum/C+epsilon);
  for(int j=0;j<C;++j)out.data[i*C+j]=spatial::f16(double(x.data[i*C+j])*double(gamma.data[j])*inv);
 }return out;
}
template<class O,int R,int C,class A>auto cast(const spatial::tensor<R,C,A>&x){
 spatial::tensor<R,C,O> out;for(int i=0;i<R*C;++i)out.data[i]=O(x.data[i]);return out;
}
}

namespace precision_experiment {
template<int M,int N>auto softmax(const spatial::tensor<M,N,spatial::f16>& x,float scale){
 spatial::tensor<M,N,float> out;
 for(int i=0;i<M;++i){
  float peak=-std::numeric_limits<float>::infinity();
  for(int j=0;j<N;++j)peak=std::max(peak,float(x.data[i*N+j])*scale);
  float sum=0;
  for(int j=0;j<N;++j){out.data[i*N+j]=std::exp(float(x.data[i*N+j])*scale-peak);sum+=out.data[i*N+j];}
  for(int j=0;j<N;++j)out.data[i*N+j]=out.data[i*N+j]/sum;
 }return out;
}
}

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
 spatial::output("__observe_input_normalized",input_normalized);

 #pragma csl dataflow rows=8 cols=8 exchange=two_hop initial_align=forward reduce=local overlap=double_buffer compute=dsr fp=relaxed
 auto q_raw=spatial::matmul(input_normalized,q_weight);
 spatial::output("__observe_q_raw",q_raw);

 #pragma csl dataflow rows=8 cols=8 exchange=two_hop initial_align=forward reduce=local overlap=double_buffer compute=dsr fp=relaxed
 auto k_raw=spatial::matmul(input_normalized,k_weight);
 spatial::output("__observe_k_raw",k_raw);

 #pragma csl dataflow rows=8 cols=8 exchange=two_hop initial_align=forward reduce=local overlap=double_buffer compute=dsr fp=relaxed
 auto v=precision_experiment::product<float>(input_normalized,v_weight);
 spatial::output("__observe_v_raw",v);

 #pragma csl dataflow rows=8 cols=8 partition=tiles coefficients=feature_pairs compute=dsd fp=relaxed
 auto q=spatial::rotate_pairs<spatial::pair_order::odd_even>(q_raw,cosine,sine);
 spatial::output("__observe_q",q);

 #pragma csl dataflow rows=8 cols=8 partition=tiles coefficients=feature_pairs compute=dsd fp=relaxed
 auto k=spatial::rotate_pairs<spatial::pair_order::odd_even>(k_raw,cosine,sine);
 spatial::output("__observe_k",k);

 auto kt=spatial::transpose(k);
 #pragma csl dataflow rows=8 cols=8 exchange=vertical_two_hop reduce=rotating_root order=east_first overlap=double_buffer compute=dsr fp=relaxed
 auto score=spatial::matmul(q,kt);
 spatial::output("__observe_score",score);

 #pragma csl dataflow rows=8 cols=8 partition=tiles reduce=max_sum accumulation=f16 math=sdk_half compute=dsr fp=relaxed elementwise=map
 auto probability=precision_experiment::softmax(score,0.125f);
 spatial::output("__observe_probability",probability);

 #pragma csl dataflow rows=8 cols=8 exchange=two_hop initial_align=both_axes reduce=local overlap=double_buffer compute=dsr fp=relaxed
 auto attention=precision_experiment::product<float>(probability,v);
 spatial::output("__observe_attention",attention);

 #pragma csl dataflow rows=8 cols=8 exchange=two_hop initial_align=forward reduce=local overlap=double_buffer compute=dsr fp=relaxed
 auto projection=precision_experiment::product<float>(attention,output_weight);
 spatial::output("__observe_projection",projection);

 #pragma csl dataflow rows=8 cols=8 partition=tiles compute=dsr fp=relaxed
 auto z=precision_experiment::add<float>(projection,input_x);
 #pragma csl dataflow rows=8 cols=8 partition=tiles reduce=bidirectional_chain weights=feature_columns accumulation=f16 math=sdk_half compute=dsr fp=relaxed
 auto x=precision_experiment::norm(z,gamma,0.000001);
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
 spatial::output("__observe_delta",delta);

 #pragma csl dataflow rows=8 cols=8 partition=tiles compute=dsr fp=relaxed
 auto result=precision_experiment::add<spatial::f16>(z,delta);
 spatial::output("output",result);
}
