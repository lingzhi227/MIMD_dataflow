#pragma once
namespace spatial {
// Unpreconditioned BiCGStab, real f32. Completed iterations include an early-s
// update; a failed denominator leaves the previous completed solution intact.
template<int N,int MaxIterations,int NNZ>
auto bicgstab_csc(const tensor<NNZ,1>& values,const index_tensor<NNZ,1>& rows,
 const index_tensor<N+1,1>& offsets,const tensor<N,1>& b,const tensor<N,1>& initial,
 const index_tensor<1,1>& limit,const tensor<2,1>& tolerances) {
 require(limit.data[0]<=MaxIterations);
 for(float v:tolerances.data) require(std::isfinite(v)&&v>=0);
 for(float v:values.data) require(std::isfinite(v));
 for(float v:b.data) require(std::isfinite(v));
 for(float v:initial.data) require(std::isfinite(v));
 auto dot=[](const tensor<N,1>& u,const tensor<N,1>& v){float q=0;for(int i=0;i<N;++i)q+=u.data[i]*v.data[i];return q;};
 solver_result<N,MaxIterations> out;out.solution=initial;
 auto ax=spmv_csc<N,N>(values,rows,offsets,initial);
 tensor<N,1> r,shadow,p,v,s;
 for(int i=0;i<N;++i) r.data[i]=b.data[i]-ax.data[i];
 shadow=r;p=r;float rho=dot(r,r);
 float alpha=1,omega=1;
 const float threshold=std::max(tolerances.data[0]*nrm2(b).data[0],tolerances.data[1]);
 out.residual_squared.data[0]=rho;
 auto reason=solver_reason::iteration_limit;
 if(!std::isfinite(rho)||!std::isfinite(threshold))reason=solver_reason::numerical_breakdown;
 else if(nrm2(r).data[0]<=threshold)reason=solver_reason::converged;
 else if(rho<=0)reason=solver_reason::numerical_breakdown;
 else for(uint32_t k=0;k<limit.data[0];++k){
  v=spmv_csc<N,N>(values,rows,offsets,p);
  const float denom=dot(shadow,v);
  if(!std::isfinite(denom)||denom==0){reason=solver_reason::numerical_breakdown;break;}
  alpha=rho/denom;
  if(!std::isfinite(alpha)){reason=solver_reason::numerical_breakdown;break;}
  for(int i=0;i<N;++i)s.data[i]=std::fma(-alpha,v.data[i],r.data[i]);
  const float ss=dot(s,s);
  if(!std::isfinite(ss)){reason=solver_reason::numerical_breakdown;break;}
  const float snorm=ss==0?nrm2(s).data[0]:std::sqrt(ss);
  if(ss==0 && snorm>threshold){reason=solver_reason::numerical_breakdown;break;}
  if(snorm<=threshold){
   for(int i=0;i<N;++i)out.solution.data[i]=std::fma(alpha,p.data[i],out.solution.data[i]);
   r=s;out.iterations.data[0]=k+1;out.residual_squared.data[k+1]=ss;reason=solver_reason::converged;break;
  }
  auto t=spmv_csc<N,N>(values,rows,offsets,s);
  const float ts=dot(t,s),tt=dot(t,t);
  if(!std::isfinite(ts)||!std::isfinite(tt)||tt<=0){reason=solver_reason::numerical_breakdown;break;}
  omega=ts/tt;
  if(!std::isfinite(omega)||omega==0){reason=solver_reason::numerical_breakdown;break;}
  for(int i=0;i<N;++i){
   out.solution.data[i]=std::fma(alpha,p.data[i],out.solution.data[i]);
   out.solution.data[i]=std::fma(omega,s.data[i],out.solution.data[i]);
   r.data[i]=std::fma(-omega,t.data[i],s.data[i]);
  }
  const float rr=dot(r,r);out.iterations.data[0]=k+1;out.residual_squared.data[k+1]=rr;
  if(!std::isfinite(rr)){reason=solver_reason::numerical_breakdown;break;}
  if(rr==0&&nrm2(r).data[0]>threshold){reason=solver_reason::numerical_breakdown;break;}
  if(std::sqrt(rr)<=threshold){reason=solver_reason::converged;break;}
  if(k+1==limit.data[0])break;
  const float next=dot(shadow,r);
  if(!std::isfinite(next)||next==0){reason=solver_reason::numerical_breakdown;break;}
  const float beta=(next/rho)*(alpha/omega);
  if(!std::isfinite(beta)){reason=solver_reason::numerical_breakdown;break;}
  for(int i=0;i<N;++i){p.data[i]=std::fma(-omega,v.data[i],p.data[i]);p.data[i]=std::fma(beta,p.data[i],r.data[i]);}
  rho=next;
 }
 ax=spmv_csc<N,N>(values,rows,offsets,out.solution);
 bool finite=true;for(int i=0;i<N;++i){r.data[i]=b.data[i]-ax.data[i];finite=finite&&std::isfinite(r.data[i]);}
 if(!finite){reason=solver_reason::numerical_breakdown;out.true_residual_norm.data[0]=std::numeric_limits<float>::infinity();}
 else{out.true_residual_norm=nrm2(r);if(reason==solver_reason::converged&&out.true_residual_norm.data[0]>threshold)reason=solver_reason::residual_gap;}
 out.reason.data[0]=static_cast<uint32_t>(reason);return out;
}
}
