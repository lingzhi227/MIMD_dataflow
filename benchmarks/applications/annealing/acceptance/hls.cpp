#include "spatial.hpp"
void design(){
auto values=spatial::input<4,3>("values");
auto result=spatial::kernel(values, [](const spatial::tensor<4,3>& values){
spatial::tensor<4,1> out{};for(int i=0;i<4;++i){float d=values.data[i*3];float t=values.data[i*3+1];float draw=values.data[i*3+2];spatial::require(t>0.0f && draw>=0.0f && draw<1.0f);if(d<0.0f || draw<spatial::exp(-d/t)){out.data[i]=1.0f;}}return out;
});spatial::output("result",result);}
