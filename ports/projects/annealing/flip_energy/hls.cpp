#include "spatial.hpp"
void design(){
auto q=spatial::input<4,4>("q");
auto spins=spatial::input<1,4>("spins");
auto result=spatial::kernel(q,spins, [](const spatial::tensor<4,4>& q,const spatial::tensor<1,4>& spins){
spatial::tensor<1,4> out{};for(int i=0;i<4;++i){spatial::require(spins.data[i]==0.0f || spins.data[i]==1.0f);float e=q.data[i*4+i];for(int j=0;j<4;++j){if(i!=j){e+=q.data[i*4+j]*spins.data[j];}}if(spins.data[i]>0.0f){e=-e;}out.data[i]=e;}return out;
});spatial::output("result",result);}
