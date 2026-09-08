#include "spatial.hpp"
// Decode source pair convention; coefficients supplied, shared across the batch.
void design() {
 auto x=spatial::input<5,1024,spatial::f16>("x");
 auto cosine=spatial::input<1,512,spatial::f16>("cosine");
 auto sine=spatial::input<1,512,spatial::f16>("sine");
#pragma csl dataflow rows=8 cols=8 partition=features axis=x layout=batch_major coefficients=feature_pairs compute=dsr fp=relaxed
 auto rotated=spatial::rotate_pairs<spatial::pair_order::odd_even>(x,cosine,sine);
 spatial::output("rotated",rotated);
}
