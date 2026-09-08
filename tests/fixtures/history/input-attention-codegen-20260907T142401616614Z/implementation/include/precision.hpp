#pragma once
namespace spatial {
// Explicit compute/storage types. The unannotated overloads retain their old
// semantics. Dataflow lowering must admit each mixed operand/transport policy.
template<class Output,class Accumulator,int R,int K,int C,class A,class B>
auto matmul(const tensor<R,K,A>&a,const tensor<K,C,B>&b){
 static_assert(std::is_same_v<Accumulator,float>,"explicit full accumulation requires f32");
 tensor<R,C,Output> out;
 for(int i=0;i<R;++i)for(int j=0;j<C;++j){
  Accumulator sum=0;
  for(int k=0;k<K;++k)sum=std::fma(Accumulator(a.data[i*K+k]),Accumulator(b.data[k*C+j]),sum);
  require(std::isfinite(sum));out.data[i*C+j]=Output(sum);
 }return out;
}
template<class Output,int R,int C,class A,class B>
auto add(const tensor<R,C,A>&a,const tensor<R,C,B>&b){
 tensor<R,C,Output> out;
 for(int i=0;i<R*C;++i){float sum=float(a.data[i])+float(b.data[i]);require(std::isfinite(sum));out.data[i]=Output(sum);}
 return out;
}
template<class Output,int M,int N,class Input>
auto softmax(const tensor<M,N,Input>&x,double scale){
 static_assert(std::is_same_v<Output,float>,"explicit wide softmax retains f32 probabilities");
 require(std::isfinite(scale)&&scale>0&&std::isfinite(float(scale)));
 tensor<M,N,Output> out;
 for(int i=0;i<M;++i){
  float peak=-std::numeric_limits<float>::infinity();
  for(int j=0;j<N;++j){float value=float(x.data[i*N+j])*float(scale);require(std::isfinite(value));peak=std::max(peak,value);}
  float sum=0;
  for(int j=0;j<N;++j){out.data[i*N+j]=std::exp(float(x.data[i*N+j])*float(scale)-peak);sum+=out.data[i*N+j];}
  require(std::isfinite(sum)&&sum>0);
  for(int j=0;j<N;++j)out.data[i*N+j]=out.data[i*N+j]/sum;
 }return out;
}
template<class Output,class Arithmetic,int M,int N,class Input,class Weight>
auto rmsnorm(const tensor<M,N,Input>&x,const tensor<1,N,Weight>&weight,double epsilon){
 static_assert(std::is_same_v<Arithmetic,float>,"explicit RMS arithmetic requires f32");
 require(std::isfinite(epsilon)&&epsilon>0&&std::isfinite(float(epsilon))&&float(epsilon)>0);
 tensor<M,N,Output> out;
 for(int i=0;i<M;++i){
  Arithmetic sum=0;
  for(int j=0;j<N;++j){Arithmetic value=Arithmetic(x.data[i*N+j]);sum=sum+value*value;}
  require(std::isfinite(sum));Arithmetic inv=Arithmetic(1)/std::sqrt(sum/Arithmetic(N)+Arithmetic(epsilon));
  for(int j=0;j<N;++j){Arithmetic weighted=Arithmetic(x.data[i*N+j])*Arithmetic(weight.data[j]);out.data[i*N+j]=Output(weighted*inv);}
 }return out;
}
}
