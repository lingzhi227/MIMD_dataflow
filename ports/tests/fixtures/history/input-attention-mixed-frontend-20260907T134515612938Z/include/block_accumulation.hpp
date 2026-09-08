#pragma once
namespace spatial {
// Explicit two-level arithmetic: binary16 fused partials over contiguous blocks,
// binary32 addition of the completed partials, then one binary16 output rounding.
// This native definition handles a short final block; a spatial backend may
// require a divisible physical partition and must reject unsupported tails.
template<class Partial,class Merge,int R,int K,int C>
auto matmul_blocked(const tensor<R,K,f16>& a,const tensor<K,C,f16>& b,int block_size) {
 static_assert(std::is_same_v<Partial,f16> && std::is_same_v<Merge,float>,
               "supported blocked arithmetic is f16 partials and f32 merge");
 require(block_size>0 && block_size<=K);
 tensor<R,C,f16> result;
 for(int i=0;i<R;++i)for(int j=0;j<C;++j){
  Merge total=0;
  for(int begin=0;begin<K;begin+=block_size){
   Partial partial=0;
   const int end=std::min(K,begin+block_size);
   for(int k=begin;k<end;++k)
    partial=Partial(std::fma(double(a.data[i*K+k]),double(b.data[k*C+j]),double(partial)));
   total=Merge(total+Merge(partial));
  }
  result.data[i*C+j]=f16(total);
 }
 return result;
}
}
