#include "spatial.hpp"
void design() {
 auto x=spatial::input<256,32>("x");
 #pragma csl dataflow rows=4 cols=4 partition=pencils exchange=sdk_transpose compute=sdk_fft result=input_layout fp=relaxed
 auto spectrum=spatial::fft3d<16,spatial::fft_direction::inverse,spatial::fft_norm::ortho>(x);
 spatial::output("spectrum",spectrum);
}
