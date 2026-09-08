#pragma once
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <limits>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <type_traits>
#include <vector>
namespace spatial {
using scalar=float;
using f16=_Float16;
inline std::unordered_map<std::string,std::vector<uint32_t>> index_inputs,index_outputs;
template<int R,int C> struct index_tensor {static_assert(R>0&&C>0);uint32_t data[R*C]{};};
template<int R,int C> auto index_input(const char*n){index_tensor<R,C>o;const auto&v=index_inputs.at(n);if(v.size()!=R*C)throw std::runtime_error("index shape");std::copy(v.begin(),v.end(),o.data);return o;}
inline std::unordered_map<std::string,std::vector<float>> inputs,outputs;
template<int R,int C,class T=scalar> struct tensor {static_assert(R>0&&C>0);static_assert(std::is_same_v<T,float>||std::is_same_v<T,f16>);T data[R*C]{};};
template<int R,int C> struct state {tensor<R,C> value;};
inline float sqrt(float x){return std::sqrt(x);} inline float exp(float x){return std::exp(x);} inline float abs(float x){return std::abs(x);}
inline void require(bool x){if(!x)throw std::runtime_error("kernel precondition");}
template<int R,int C,class T=scalar> tensor<R,C,T> input(const char*n){tensor<R,C,T>o;const auto&v=inputs.at(n);if(v.size()!=R*C)throw std::runtime_error("shape");std::copy(v.begin(),v.end(),o.data);return o;}
// Optional executable magnitude precondition; also retained in the typed IR.
template<int R,int C,class T=scalar> tensor<R,C,T> input(const char*n,double abs_bound){
 require(std::isfinite(abs_bound)&&abs_bound>=0);auto out=input<R,C,T>(n);
 for(const auto value:out.data)require(std::isfinite(double(value))&&std::abs(double(value))<=abs_bound);
 return out;
}
template<int R,int C,class T> void output(const char*n,const tensor<R,C,T>&a){if(outputs.count(n)||index_outputs.count(n))throw std::runtime_error("duplicate output");outputs[n]=std::vector<float>(a.data,a.data+R*C);}
template<int R,int C> void output(const char*n,const index_tensor<R,C>&a){if(outputs.count(n)||index_outputs.count(n))throw std::runtime_error("duplicate output");index_outputs[n]=std::vector<uint32_t>(a.data,a.data+R*C);}
template<int R,int C,class F> auto map(const tensor<R,C>&a,F f){tensor<R,C>o;for(int i=0;i<R*C;++i)o.data[i]=f(a.data[i]);return o;}
template<int R,int C,class T> auto add(const tensor<R,C,T>&a,const tensor<R,C,T>&b){
 tensor<R,C,T>o;
 for(int i=0;i<R*C;++i){
  if constexpr(std::is_same_v<T,f16>){o.data[i]=static_cast<f16>(double(a.data[i])+double(b.data[i]));}
  else{o.data[i]=a.data[i]+b.data[i];}
 }return o;
}
template<int R,int C,class T> auto transpose(const tensor<R,C,T>&a){tensor<C,R,T>o;for(int i=0;i<R;++i)for(int j=0;j<C;++j)o.data[j*R+i]=a.data[i*C+j];return o;}
template<int R,int K,int C,class T> auto matmul(const tensor<R,K,T>&a,const tensor<K,C,T>&b){
 tensor<R,C,T>o;
 for(int i=0;i<R;++i)for(int j=0;j<C;++j)for(int k=0;k<K;++k){
  if constexpr(std::is_same_v<T,f16>){
   // One binary16 rounding per fused update; matches the target arithmetic probe.
   o.data[i*C+j]=static_cast<f16>(std::fma(double(a.data[i*K+k]),double(b.data[k*C+j]),double(o.data[i*C+j])));
  }else{o.data[i*C+j]+=a.data[i*K+k]*b.data[k*C+j];}
 }return o;
}
// Distributed reductions retain ordinary C++ numerical meaning.
template<int N> auto dot(const tensor<N,1>&x,const tensor<N,1>&y){
 tensor<1,1> out;
 for(int i=0;i<N;++i){require(std::isfinite(x.data[i])&&std::isfinite(y.data[i]));out.data[0]+=x.data[i]*y.data[i];}
 return out;
}
template<int N> auto nrm2(const tensor<N,1>&x){
 tensor<1,1> out;float peak=0;
 for(int i=0;i<N;++i){require(std::isfinite(x.data[i]));peak=std::max(peak,std::abs(x.data[i]));}
 if(peak==0)return out;
 int exponent;std::frexp(peak,&exponent);
 float alpha=std::ldexp(1.0f,std::max(-126,exponent-1));
 float inv=1.0f/alpha,sum=0;
 for(int i=0;i<N;++i){float v=x.data[i]*inv;sum+=v*v;}
 out.data[0]=std::sqrt(sum)*alpha;return out;
}
// Canonical CSC: unsigned indices, sorted unique rows, explicit zero entries.
template<int M,int N,int NNZ> auto spmv_csc(const tensor<NNZ,1>&values,const index_tensor<NNZ,1>&rows,const index_tensor<N+1,1>&offsets,const tensor<N,1>&x){
 tensor<M,1> y;
 require(offsets.data[0]==0 && offsets.data[N]==NNZ);
 for(int col=0;col<N;++col){
  require(offsets.data[col]<=offsets.data[col+1] && offsets.data[col+1]<=NNZ);
  for(uint32_t p=offsets.data[col];p<offsets.data[col+1];++p){
   require(rows.data[p]<M && (p==offsets.data[col] || rows.data[p-1]<rows.data[p]));
   y.data[rows.data[p]]+=values.data[p]*x.data[col];
  }
 }return y;
}
// R-only QR; adjacent Givens reference, row signs unspecified by spatial profile.
template<int M,int N> auto qr_r(const tensor<M,N>&a){
 static_assert(M>=N);
 auto o=a;float scale=0;
 for(int i=0;i<M*N;++i){require(std::isfinite(a.data[i]));scale=std::max(scale,std::abs(a.data[i]));}
 require(scale>0);
 for(int col=0;col<N;++col){
  for(int row=M-1;row>col;--row){
   float x=o.data[(row-1)*N+col],y=o.data[row*N+col],c,s,t;
   if(y==0){c=1;s=0;}
   else if(std::abs(y)>std::abs(x)){t=-x/y;s=1.0f/std::sqrt(1.0f+t*t);c=s*t;}
   else {t=-y/x;c=1.0f/std::sqrt(1.0f+t*t);s=c*t;}
   for(int j=col;j<N;++j){float u=o.data[(row-1)*N+j],v=o.data[row*N+j];
    o.data[(row-1)*N+j]=c*u-s*v;o.data[row*N+j]=s*u+c*v;
   }
  }
  require(std::abs(o.data[col*N+col])>0.00000762939453125f*scale);
 }
 return o;
}
// Packed no-pivot LU; first spatial profile requires strictly row-dominant input.
template<int N> auto lu_no_pivot(const tensor<N,N>&a){
 auto o=a;
 for(int i=0;i<N;++i){double off=0;
  for(int j=0;j<N;++j){require(std::isfinite(a.data[i*N+j]));if(i!=j)off+=std::abs(double(a.data[i*N+j]));}
  require(std::abs(double(a.data[i*N+i]))>off);
 }
 for(int k=0;k<N;++k){
  require(std::isfinite(o.data[k*N+k]) && o.data[k*N+k]!=0);
  for(int i=k+1;i<N;++i)o.data[i*N+k]/=o.data[k*N+k];
  for(int i=k+1;i<N;++i)for(int j=k+1;j<N;++j)o.data[i*N+j]-=o.data[i*N+k]*o.data[k*N+j];
 }
 return o;
}
// Right-looking lower Cholesky; symmetric positive definite input is required.
template<int N> auto cholesky(const tensor<N,N>&a){
 auto o=a;
 for(int i=0;i<N;++i)for(int j=0;j<N;++j)
  require(std::isfinite(a.data[i*N+j]) && a.data[i*N+j]==a.data[j*N+i]);
 for(int i=0;i<N;++i)for(int j=i+1;j<N;++j)o.data[i*N+j]=0;
 for(int k=0;k<N;++k){
  require(std::isfinite(o.data[k*N+k]) && o.data[k*N+k]>0);
  o.data[k*N+k]=std::sqrt(o.data[k*N+k]);
  for(int i=k+1;i<N;++i)o.data[i*N+k]/=o.data[k*N+k];
  for(int i=k+1;i<N;++i)for(int j=k+1;j<=i;++j)
   o.data[i*N+j]-=o.data[i*N+k]*o.data[j*N+k];
 }
 return o;
}
template<int R,int C> auto row_sum(const tensor<R,C>&a){tensor<R,1>o;for(int i=0;i<R;++i)for(int j=0;j<C;++j)o.data[i]+=a.data[i*C+j];return o;}
template<int R,int C> auto accumulate(state<R,C>&s,const tensor<R,C>&a){s.value=add(s.value,a);return s.value;}
template<class A,class F> auto kernel(const A&a,F f){return f(a);}
template<class A,class B,class F> auto kernel(const A&a,const B&b,F f){return f(a,b);}
template<int X,int Y,int Z,int Steps,int C,class F>
auto grid_iterate(const tensor<X*Y,Z>&initial,const tensor<1,C>&coeff,F update){
 auto current=initial;
 for(int step=0;step<Steps;++step){tensor<X*Y,Z> next{};
  for(int x=0;x<X;++x)for(int y=0;y<Y;++y){tensor<7,Z> n{};
   for(int k=0;k<Z;++k){
    if(x>0)n.data[k]=current.data[((x-1)*Y+y)*Z+k];
    if(x+1<X)n.data[Z+k]=current.data[((x+1)*Y+y)*Z+k];
    if(y>0)n.data[2*Z+k]=current.data[(x*Y+y-1)*Z+k];
    if(y+1<Y)n.data[3*Z+k]=current.data[(x*Y+y+1)*Z+k];
    if(k>0)n.data[4*Z+k]=current.data[(x*Y+y)*Z+k-1];
    if(k+1<Z)n.data[5*Z+k]=current.data[(x*Y+y)*Z+k+1];
    n.data[6*Z+k]=current.data[(x*Y+y)*Z+k];
   }
   auto value=update(n,coeff);for(int k=0;k<Z;++k)next.data[(x*Y+y)*Z+k]=value.data[k];
  }current=next;
 }return current;
}
}

#include "solver.hpp"

#include "bicgstab.hpp"

#include "power.hpp"

#include "fft.hpp"

#include "normalization.hpp"

#include "activation.hpp"

#include "pair_rotation.hpp"

#include "block_accumulation.hpp"

#include "precision.hpp"
