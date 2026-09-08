#pragma once
#include <complex>
namespace spatial {
enum class fft_direction { forward, inverse };
enum class fft_norm { backward, ortho, forward };
// Numerical reference: radix-2 complex transform; spatial scheduling is a lowering policy.
// Double intermediates deliberately provide an independent accuracy reference, not target bits.
template<int N, fft_direction Direction, fft_norm Norm>
auto fft3d(const tensor<N*N,2*N>& input) {
 static_assert(N>=4 && (N&(N-1))==0);
 std::vector<std::complex<double>> data(N*N*N), line(N);
 for(int i=0;i<N*N*N;++i){require(std::isfinite(input.data[2*i])&&std::isfinite(input.data[2*i+1]));data[i]={input.data[2*i],input.data[2*i+1]};}
 auto transform=[&](){
  for(int i=1,j=0;i<N;++i){int bit=N>>1;for(;j&bit;bit>>=1)j^=bit;j^=bit;if(i<j)std::swap(line[i],line[j]);}
  for(int len=2;len<=N;len*=2)for(int start=0;start<N;start+=len)for(int j=0;j<len/2;++j){
   double angle=(Direction==fft_direction::inverse?2.0:-2.0)*std::acos(-1.0)*j/len;
   auto u=line[start+j],v=line[start+j+len/2]*std::complex<double>(std::cos(angle),std::sin(angle));
   line[start+j]=u+v;line[start+j+len/2]=u-v;
  }
 };
 for(int axis=2;axis>=0;--axis)for(int a=0;a<N;++a)for(int b=0;b<N;++b){
  auto index=[&](int k){return axis==2?(a*N+b)*N+k:axis==1?(a*N+k)*N+b:(k*N+a)*N+b;};
  for(int k=0;k<N;++k)line[k]=data[index(k)];transform();for(int k=0;k<N;++k)data[index(k)]=line[k];
 }
 double scale=1;
 if constexpr(Norm==fft_norm::ortho)scale=1/std::sqrt(double(N)*N*N);
 else if constexpr((Norm==fft_norm::backward && Direction==fft_direction::inverse)||(Norm==fft_norm::forward && Direction==fft_direction::forward))scale=1/(double(N)*N*N);
 tensor<N*N,2*N> out;
 for(int i=0;i<N*N*N;++i){out.data[2*i]=float(data[i].real()*scale);out.data[2*i+1]=float(data[i].imag()*scale);}return out;
}
}
