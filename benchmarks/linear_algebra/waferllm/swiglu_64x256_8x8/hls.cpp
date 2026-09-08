#include "spatial.hpp"
void design() {
 auto up=spatial::input<64,256,spatial::f16>("up");
 auto gate=spatial::input<64,256,spatial::f16>("gate");
 #pragma csl dataflow rows=8 cols=8 partition=tiles math=sdk_half compute=dsr elementwise=map fp=relaxed
 auto activated=spatial::silu(gate);
 #pragma csl dataflow rows=8 cols=8 partition=tiles math=sdk_half compute=dsr elementwise=map fp=relaxed
 auto gated=spatial::multiply(up,activated);
 spatial::output("gated",gated);
}
