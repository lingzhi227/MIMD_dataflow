#include "spatial.hpp"
void design() {
 auto x=spatial::input<4096,128>("x");
 #pragma csl dataflow rows=16 cols=16 partition=pencils exchange=sdk_transpose compute=sdk_fft result=input_layout fp=relaxed
 auto spectrum=spatial::fft3d<64,spatial::fft_direction::forward,spatial::fft_norm::backward>(x);
 spatial::output("spectrum",spectrum);
}
