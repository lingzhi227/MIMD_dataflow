#include "spatial.hpp"
void design(){
 auto field=spatial::input<4,32>("field");
 auto coeff=spatial::input<1,7>("coeff");
 #pragma csl vectorize
 auto result=spatial::grid_iterate<2,2,32,4>(field,coeff,[](const spatial::tensor<7,32>&n,const spatial::tensor<1,7>&c){
  spatial::tensor<1,32> out{};
  for(int k=0;k<32;++k){
   float value=c.data[6]*n.data[192+k];
   for(int direction=0;direction<6;++direction){value+=c.data[direction]*n.data[direction*32+k];}
   out.data[k]=value;
  }return out;
 });
 spatial::output("result",result);
}
