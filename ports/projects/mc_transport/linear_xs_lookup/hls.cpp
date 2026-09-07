#include "spatial.hpp"
void design(){
auto particles=spatial::input<4,6>("particles");
auto table=spatial::input<1,50>("table");
auto result=spatial::kernel(particles,table, [](const spatial::tensor<4,6>& particles,const spatial::tensor<1,50>& table){
spatial::tensor<4,6> out=particles;
for(int p=0;p<4;++p){float energy=particles.data[p*6];for(int n=0;n<2;++n){
 spatial::require(energy>=table.data[n*4] && energy<=table.data[n*4+3]);
 int lower=n*4;
 for(int j=n*4;j<n*4+3;++j){if(energy>=table.data[j]){lower=j;}}
 float lo=table.data[lower];float hi=table.data[lower+1];spatial::require(hi>lo);
 float f=(hi-energy)/(hi-lo);
 for(int xs=0;xs<5;++xs){float low=table.data[8+lower*5+xs];float high=table.data[8+(lower+1)*5+xs];out.data[p*6+1+xs]+=table.data[48+n]*(high-f*(high-low));}
}}return out;
});spatial::output("result",result);}
