#include "spatial.hpp"
void design(){
 auto field=spatial::input<1,7>("field");
 auto coeff=spatial::input<1,7>("coeff");
 
 auto result=spatial::grid_iterate<1,1,7,3>(field,coeff,[](const spatial::tensor<7,7>&n,const spatial::tensor<1,7>&c){
  spatial::tensor<1,7> out{};
  for(int k=0;k<7;++k){
   float value=c.data[6]*n.data[42+k];
   for(int direction=0;direction<6;++direction){value+=c.data[direction]*n.data[direction*7+k];}
   out.data[k]=value;
  }return out;
 });
 spatial::output("result",result);
}
