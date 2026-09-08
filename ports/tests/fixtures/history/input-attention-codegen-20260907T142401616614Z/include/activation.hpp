#pragma once
namespace spatial {
// Stable standard-math native reference. The device approximation/range is an
// explicit dataflow policy; the intermediate activation is stored in binary16.
template<int R,int C> auto silu(const tensor<R,C,f16>& x) {
 tensor<R,C,f16> result;
 for(int i=0;i<R*C;++i){
  const double value=x.data[i];require(std::isfinite(value));
  const double e=std::exp(-std::abs(value));
  result.data[i]=f16(value>=0 ? value/(1+e) : value*e/(1+e));
 }
 return result;
}
template<int R,int C,class T> auto multiply(const tensor<R,C,T>& a,const tensor<R,C,T>& b){
 tensor<R,C,T> result;
 for(int i=0;i<R*C;++i)result.data[i]=T(double(a.data[i])*double(b.data[i]));
 return result;
}
}
