#include "spatial.hpp"
void design(){auto samples=spatial::input<1,6>("samples");auto result=spatial::kernel(samples,[](const spatial::tensor<1,6>& samples){
spatial::tensor<1,1> out{};
float s0=1.125000000e+02f;
float s1=samples.data[0];
float s2=samples.data[1];
float s3=s1+s2;
float s4=samples.data[2];
float s5=s3*s4;
float s6=samples.data[3];
float s7=samples.data[4];
float s8=s6+s7;
float s9=s8-s5;
float s10=s0*s9;
float s11=samples.data[5];
float s12=s11*s10;
out.data[0]=s12;
return out;
});spatial::output("result",result);}
