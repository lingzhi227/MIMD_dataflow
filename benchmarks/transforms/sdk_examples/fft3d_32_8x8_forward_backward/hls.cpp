#include "spatial.hpp"
void design() {
 auto x=spatial::input<1024,64>("x");
 #pragma csl dataflow rows=8 cols=8 partition=pencils exchange=sdk_transpose compute=sdk_fft result=input_layout fp=relaxed
 auto spectrum=spatial::fft3d<32,spatial::fft_direction::forward,spatial::fft_norm::backward>(x);
 spatial::output("spectrum",spectrum);
}
