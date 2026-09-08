#include "spatial.hpp"
void design(){auto a=spatial::input<4,4>("a");auto b=spatial::input<4,1>("b");auto result=spatial::kernel(a,b,[](const spatial::tensor<4,4>&a,const spatial::tensor<4,1>&b){
spatial::tensor<4,1> x{};spatial::tensor<4,1> r=b;spatial::tensor<4,1> p{};spatial::tensor<4,1> v{};spatial::tensor<4,1> s{};spatial::tensor<4,1> t{};
float old=1.0f;float alpha=1.0f;float omega=1.0f;
for(int iter=0;iter<16;++iter){float rr=0.0f;for(int i=0;i<4;++i){rr+=r.data[i]*r.data[i];}if(rr>0.000000000001f){
 float rho=0.0f;for(int i=0;i<4;++i){rho+=b.data[i]*r.data[i];}spatial::require(rho!=0.0f && omega!=0.0f);float beta=(rho/old)*(alpha/omega);
 for(int i=0;i<4;++i){p.data[i]=r.data[i]+beta*(p.data[i]-omega*v.data[i]);}
 for(int i=0;i<4;++i){v.data[i]=0.0f;for(int j=0;j<4;++j){v.data[i]+=a.data[i*4+j]*p.data[j];}}
 float bv=0.0f;for(int i=0;i<4;++i){bv+=b.data[i]*v.data[i];}spatial::require(bv!=0.0f);alpha=rho/bv;float ss=0.0f;
 for(int i=0;i<4;++i){s.data[i]=r.data[i]-alpha*v.data[i];ss+=s.data[i]*s.data[i];}
 if(ss<0.000000000001f){for(int i=0;i<4;++i){x.data[i]+=alpha*p.data[i];r.data[i]=s.data[i];}}
 else{for(int i=0;i<4;++i){t.data[i]=0.0f;for(int j=0;j<4;++j){t.data[i]+=a.data[i*4+j]*s.data[j];}}
 float ts=0.0f;float tt=0.0f;for(int i=0;i<4;++i){ts+=t.data[i]*s.data[i];tt+=t.data[i]*t.data[i];}spatial::require(tt>0.0f);omega=ts/tt;
 for(int i=0;i<4;++i){x.data[i]+=alpha*p.data[i]+omega*s.data[i];r.data[i]=s.data[i]-omega*t.data[i];}}
 old=rho;
}}return x;
});spatial::output("result",result);}
