#pragma once
namespace spatial {
// High-precision native algorithm reference; device accumulation/math are explicit
// lowering policies. Weight j always belongs to feature j, normalization to row i.
template<int M,int N> auto rmsnorm(const tensor<M,N,f16>& x,
                                  const tensor<1,N,f16>& weight, double epsilon) {
 require(std::isfinite(epsilon) && epsilon>0);
 tensor<M,N,f16> out;
 for(int i=0;i<M;++i){
  double sum=0;
  for(int j=0;j<N;++j){double v=x.data[i*N+j];require(std::isfinite(v));sum+=v*v;}
  double inverse=1/std::sqrt(sum/N+epsilon);
  for(int j=0;j<N;++j){require(std::isfinite(double(weight.data[j])));out.data[i*N+j]=f16(double(x.data[i*N+j])*double(weight.data[j])*inverse);}
 }
 return out;
}
}

namespace spatial {
// Numerically stable reference, including all-negative and constant score rows.
template<int M,int N> auto softmax(const tensor<M,N,f16>& x,double scale) {
 require(std::isfinite(scale) && scale>0);
 tensor<M,N,f16> out;
 for(int i=0;i<M;++i){
  double peak=-std::numeric_limits<double>::infinity(),sum=0;
  for(int j=0;j<N;++j){require(std::isfinite(double(x.data[i*N+j])));peak=std::max(peak,double(x.data[i*N+j])*scale);}
  for(int j=0;j<N;++j)sum+=std::exp(double(x.data[i*N+j])*scale-peak);
  for(int j=0;j<N;++j)out.data[i*N+j]=f16(std::exp(double(x.data[i*N+j])*scale-peak)/sum);
 }
 return out;
}
}
