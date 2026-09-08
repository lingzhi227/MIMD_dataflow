#include "spatial.hpp"
void design(){
auto a=spatial::input<4,4>("a");
auto b=spatial::input<4,1>("b");
auto result=spatial::kernel(a,b, [](const spatial::tensor<4,4>& a,const spatial::tensor<4,1>& b){
spatial::tensor<4,1> x{};spatial::tensor<4,1> r=b;spatial::tensor<4,1> p=b;spatial::tensor<4,1> w{};
float rr=0.0f;for(int i=0;i<4;++i){rr+=r.data[i]*r.data[i];}
for(int iter=0;iter<8;++iter){if(rr>0.000000000001f){
 for(int i=0;i<4;++i){w.data[i]=0.0f;for(int j=0;j<4;++j){w.data[i]+=a.data[i*4+j]*p.data[j];}}
 float pw=0.0f;for(int i=0;i<4;++i){pw+=p.data[i]*w.data[i];}spatial::require(pw>0.0f);float alpha=rr/pw;
 for(int i=0;i<4;++i){x.data[i]+=alpha*p.data[i];r.data[i]-=alpha*w.data[i];}
 float next=0.0f;for(int i=0;i<4;++i){next+=r.data[i]*r.data[i];}float beta=next/rr;for(int i=0;i<4;++i){p.data[i]=r.data[i]+beta*p.data[i];}rr=next;
}}return x;
});spatial::output("result",result);}
