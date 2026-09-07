#include "spatial.hpp"
void design(){auto samples=spatial::input<1,6>("samples");auto result=spatial::kernel(samples,[](const spatial::tensor<1,6>& samples){
spatial::tensor<1,1> out{};
float s0=1.666600000e-01f;
float s1=samples.data[0];
float s2=samples.data[1];
float s3=samples.data[2];
float s4=samples.data[3];
float s5=samples.data[4];
float s6=samples.data[5];
float s7=s6+s5;
float s8=s7+s4;
float s9=s8+s3;
float s10=s9+s2;
float s11=s10+s1;
float s12=s11*s0;
out.data[0]=s12;
return out;
});spatial::output("result",result);}
