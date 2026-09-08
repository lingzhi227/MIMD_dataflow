#include "spatial.hpp"
// Frontend boundary fixture, not a qualified complete application.
void design() {
 auto z=spatial::input<3,256,spatial::f16>("z",34.5625);
 auto gamma=spatial::input<1,256,spatial::f16>("gamma",1.0);
#pragma csl dataflow rows=16 cols=16 partition=features axis=y layout=batch_major reduce=sdk_axis result=replicated_columns accumulation=f16 collective=f32 statistic=mean math=sdk_half compute=dsr fp=relaxed
 auto normalized=spatial::rmsnorm(z,gamma,0.000001);
 spatial::output("normalized",normalized);
}
