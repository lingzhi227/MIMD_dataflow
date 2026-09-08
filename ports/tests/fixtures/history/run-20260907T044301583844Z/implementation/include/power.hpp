#pragma once
namespace spatial {
enum class power_reason:uint32_t {completed=0,zero_norm=1,numerical_breakdown=2};
template<int N,int MaxIterations> struct power_result {
 static_assert(N>0 && MaxIterations>0);
 tensor<N,1> vector;
 index_tensor<1,1> reason;
 index_tensor<1,1> iterations;
 tensor<MaxIterations,1> norms;
};
// SDK-style fixed-step normalized power iteration. Zero steps returns the
// supplied vector exactly; completion does not certify eigenpair convergence.
template<int N,int MaxIterations,int NNZ>
auto power_csc(const tensor<NNZ,1>& values,const index_tensor<NNZ,1>& rows,
 const index_tensor<N+1,1>& offsets,const tensor<N,1>& initial,
 const index_tensor<1,1>& steps) {
 require(steps.data[0]<=MaxIterations);
 for(float v:values.data)require(std::isfinite(v));
 for(float v:initial.data)require(std::isfinite(v));
 power_result<N,MaxIterations> out;out.vector=initial;
 // Validate canonical CSC even for a zero-step invocation.
 (void)spmv_csc<N,N>(values,rows,offsets,initial);
 for(uint32_t k=0;k<steps.data[0];++k){
  auto y=spmv_csc<N,N>(values,rows,offsets,out.vector);
  bool finite=true;for(float v:y.data)finite=finite&&std::isfinite(v);
  if(!finite){out.reason.data[0]=uint32_t(power_reason::numerical_breakdown);break;}
  const float norm=nrm2(y).data[0];out.norms.data[k]=norm;
  if(norm==0){out.reason.data[0]=uint32_t(power_reason::zero_norm);break;}
  const float inverse=1.0f/norm;
  if(!std::isfinite(norm)||!std::isfinite(inverse)){out.reason.data[0]=uint32_t(power_reason::numerical_breakdown);break;}
  for(int i=0;i<N;++i)out.vector.data[i]=y.data[i]*inverse;
  out.iterations.data[0]=k+1;
 }
 return out;
}
}
