#pragma once
namespace spatial {
enum class pair_order {even_odd, odd_even};
// Adjacent feature pairs; coefficient rows are explicit token rows or broadcast.
// odd_even first permutes each input pair, then applies the usual 2D rotation.
template<pair_order Order,int M,int N,int C> auto rotate_pairs(
 const tensor<M,N,f16>& x,const tensor<C,N/2,f16>& cosine,
 const tensor<C,N/2,f16>& sine) {
 static_assert(N%2==0 && (C==1 || C==M));
 tensor<M,N,f16> out;
 for(int i=0;i<M;++i)for(int j=0;j<N/2;++j){
  double a=x.data[i*N+2*j+(Order==pair_order::odd_even?1:0)];
  double b=x.data[i*N+2*j+(Order==pair_order::odd_even?0:1)];
  double c=cosine.data[(C==1?0:i)*(N/2)+j],s=sine.data[(C==1?0:i)*(N/2)+j];
  require(std::isfinite(a)&&std::isfinite(b)&&std::isfinite(c)&&std::isfinite(s));
  out.data[i*N+2*j]=f16(a*c-b*s);out.data[i*N+2*j+1]=f16(b*c+a*s);
 }
 return out;
}
}
