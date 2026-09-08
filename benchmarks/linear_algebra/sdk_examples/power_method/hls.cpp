#include "spatial.hpp"
void design(){auto a=spatial::input<4,4>("a");auto b=spatial::input<4,1>("b");auto result=spatial::kernel(a,b,[](const spatial::tensor<4,4>&a,const spatial::tensor<4,1>&b){
spatial::tensor<4,1> x=b;spatial::tensor<4,1> y{};
for(int iter=0;iter<8;++iter){for(int i=0;i<4;++i){y.data[i]=0.0f;for(int j=0;j<4;++j){y.data[i]+=a.data[i*4+j]*x.data[j];}}
 float norm=0.0f;for(int i=0;i<4;++i){norm+=y.data[i]*y.data[i];}spatial::require(norm>0.0f);float inv=1.0f/spatial::sqrt(norm);for(int i=0;i<4;++i){x.data[i]=y.data[i]*inv;}}
return x;
});spatial::output("result",result);}
