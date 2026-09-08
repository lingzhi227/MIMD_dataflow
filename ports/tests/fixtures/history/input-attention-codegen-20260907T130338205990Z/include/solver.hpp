#pragma once
// Included after the tensor/CSC primitives in spatial.hpp.
namespace spatial {
enum class solver_reason : uint32_t {
  converged = 0,
  iteration_limit = 1,
  nonpositive_curvature = 2,
  numerical_breakdown = 3,
  residual_gap = 4
};
template<int N, int MaxIterations> struct solver_result {
  static_assert(N > 0 && MaxIterations > 0);
  tensor<N,1> solution;
  index_tensor<1,1> reason;
  index_tensor<1,1> iterations;
  tensor<MaxIterations+1,1> residual_squared;
  tensor<1,1> true_residual_norm;
};
// Ordinary C++ reference semantics. The spatial lowering may reassociate sparse
// products and reductions, while preserving termination reasons and true-residual
// validation. Tolerances are [relative, absolute]; the iteration limit is u32.
// Positive definiteness is a caller assumption, not an eigenvalue test here.
template<bool Jacobi, int N, int MaxIterations, int NNZ>
auto krylov_csc(const tensor<NNZ,1>& values,
            const index_tensor<NNZ,1>& rows,
            const index_tensor<N+1,1>& offsets,
            const tensor<N,1>& b,
            const tensor<N,1>& initial,
            const index_tensor<1,1>& limit,
            const tensor<2,1>& tolerances) {
  require(limit.data[0] <= MaxIterations);
  for(float v: tolerances.data) require(std::isfinite(v) && v >= 0);
  for(float v: values.data) require(std::isfinite(v));
  for(float v: b.data) require(std::isfinite(v));
  for(float v: initial.data) require(std::isfinite(v));
  const auto product=[](const tensor<N,1>& u,const tensor<N,1>& v) {
    float sum=0;for(int i=0;i<N;++i) sum+=u.data[i]*v.data[i];return sum;
  };
  solver_result<N,MaxIterations> out;
  out.solution = initial;
  auto ax = spmv_csc<N,N>(values,rows,offsets,initial);
  tensor<N,1> r, p, inverse, z;
  if constexpr(Jacobi) {
    for(int c=0;c<N;++c) {
      float diag=0;
      for(uint32_t q=offsets.data[c];q<offsets.data[c+1];++q)
        if(rows.data[q]==uint32_t(c)) diag=values.data[q];
      require(diag>=0x1p-16f); // bounded positive Jacobi input domain
      inverse.data[c]=1.0f/diag;
    }
  }
  for(int i=0;i<N;++i) r.data[i]=b.data[i]-ax.data[i];
  p=r;
  float rho=product(r,r);
  const float threshold=std::max(tolerances.data[0]*nrm2(b).data[0],tolerances.data[1]);
  auto reason=solver_reason::iteration_limit;
  out.residual_squared.data[0]=rho;
  if(!std::isfinite(rho) || !std::isfinite(threshold)) reason=solver_reason::numerical_breakdown;
  else if(nrm2(r).data[0] <= threshold) reason=solver_reason::converged;
  else if(rho <= 0) reason=solver_reason::numerical_breakdown;
  else {
    for(uint32_t k=0;k<limit.data[0];++k) {
      if constexpr(Jacobi) {
        for(int i=0;i<N;++i) z.data[i]=inverse.data[i]*r.data[i];
        const float weighted=product(r,z);
        if(!std::isfinite(weighted) || weighted<=0) {reason=solver_reason::numerical_breakdown;break;}
        if(k==0) p=z;
        else {
          const float beta=weighted/rho;
          if(!std::isfinite(beta)) {reason=solver_reason::numerical_breakdown;break;}
          for(int i=0;i<N;++i) p.data[i]=std::fma(beta,p.data[i],z.data[i]);
        }
        rho=weighted;
      }
      auto ap=spmv_csc<N,N>(values,rows,offsets,p);
      const float curvature=product(p,ap);
      if(!std::isfinite(curvature)) {reason=solver_reason::numerical_breakdown;break;}
      if(curvature<=0) {reason=solver_reason::nonpositive_curvature;break;}
      const float alpha=rho/curvature;
      if(!std::isfinite(alpha)) {reason=solver_reason::numerical_breakdown;break;}
      for(int i=0;i<N;++i) {
        out.solution.data[i]=std::fma(alpha,p.data[i],out.solution.data[i]);
        r.data[i]=std::fma(-alpha,ap.data[i],r.data[i]);
      }
      const float next=product(r,r);
      out.iterations.data[0]=k+1;
      out.residual_squared.data[k+1]=next;
      if(!std::isfinite(next)) {reason=solver_reason::numerical_breakdown;break;}
      if(next==0 && nrm2(r).data[0]>threshold) {reason=solver_reason::numerical_breakdown;break;}
      if(std::sqrt(next)<=threshold) {reason=solver_reason::converged;break;}
      // No next search direction is needed after the final permitted update.
      if(k+1==limit.data[0]) break;
      if constexpr(!Jacobi) {
        const float beta=next/rho;
        if(!std::isfinite(beta)) {reason=solver_reason::numerical_breakdown;break;}
        for(int i=0;i<N;++i) p.data[i]=std::fma(beta,p.data[i],r.data[i]);
        rho=next;
      }
    }
  }
  // Do not equate a small recursively updated residual with the true residual.
  ax=spmv_csc<N,N>(values,rows,offsets,out.solution);
  for(int i=0;i<N;++i) r.data[i]=b.data[i]-ax.data[i];
  bool finite=true;
  for(float v:r.data) finite=finite && std::isfinite(v);
  if(!finite) {
    reason=solver_reason::numerical_breakdown;
    out.true_residual_norm.data[0]=std::numeric_limits<float>::infinity();
  } else {
    out.true_residual_norm=nrm2(r);
    if(reason==solver_reason::converged && out.true_residual_norm.data[0]>threshold)
      reason=solver_reason::residual_gap;
  }
  out.reason.data[0]=static_cast<uint32_t>(reason);
  return out;
}
template<int N,int MaxIterations,int NNZ>
auto cg_csc(const tensor<NNZ,1>& values,const index_tensor<NNZ,1>& rows,
 const index_tensor<N+1,1>& offsets,const tensor<N,1>& b,const tensor<N,1>& initial,
 const index_tensor<1,1>& limit,const tensor<2,1>& tolerances) {
 return krylov_csc<false,N,MaxIterations>(values,rows,offsets,b,initial,limit,tolerances);
}
template<int N,int MaxIterations,int NNZ>
auto pcg_csc(const tensor<NNZ,1>& values,const index_tensor<NNZ,1>& rows,
 const index_tensor<N+1,1>& offsets,const tensor<N,1>& b,const tensor<N,1>& initial,
 const index_tensor<1,1>& limit,const tensor<2,1>& tolerances) {
 return krylov_csc<true,N,MaxIterations>(values,rows,offsets,b,initial,limit,tolerances);
}
}
