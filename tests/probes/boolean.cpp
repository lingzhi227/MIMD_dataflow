#include "spatial.hpp"
void design(){
 auto a=spatial::input<1,1>("a");
 auto result=spatial::kernel(a,[](const spatial::tensor<1,1>&a){
  spatial::tensor<1,1> out{};
  bool enabled=true;
  if(enabled && !(a.data[0]<0.0f)){out.data[0]=+a.data[0];}
  return out;
 });
 #pragma csl resident
 static spatial::state<1,1> sum;
 auto accumulated=spatial::accumulate(sum,result);
 spatial::output("result",accumulated);
}
