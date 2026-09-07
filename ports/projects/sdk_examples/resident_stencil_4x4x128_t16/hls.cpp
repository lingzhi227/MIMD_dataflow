#include "spatial.hpp"
void design(){
 auto field=spatial::input<16,128>("field");
 auto coeff=spatial::input<1,7>("coeff");
 #pragma csl vectorize
 auto result=spatial::grid_iterate<4,4,128,16>(field,coeff,[](const spatial::tensor<7,128>&n,const spatial::tensor<1,7>&c){
  spatial::tensor<1,128> out{};
  for(int k=0;k<128;++k){
   float value=c.data[6]*n.data[768+k];
   for(int direction=0;direction<6;++direction){value+=c.data[direction]*n.data[direction*128+k];}
   out.data[k]=value;
  }return out;
 });
 spatial::output("result",result);
}
