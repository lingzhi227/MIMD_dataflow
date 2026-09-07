#include "spatial.hpp"
void design() {
  auto a=spatial::input<4,4>("a");
  auto result=spatial::kernel(a, [](const spatial::tensor<4,4>& a) {
spatial::tensor<4,4> lu=a;
for(int k=0;k<4;++k){
 spatial::require(spatial::abs(lu.data[k*4+k])>0.00001f);
 for(int i=k+1;i<4;++i){
  lu.data[i*4+k]/=lu.data[k*4+k];
  for(int j=k+1;j<4;++j){lu.data[i*4+j]-=lu.data[i*4+k]*lu.data[k*4+j];}
 }
}
return lu;
  });
  spatial::output("result",result);
}
