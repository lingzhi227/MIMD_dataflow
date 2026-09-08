#include "spatial.hpp"
void design() {
 auto x=spatial::input<64,128,spatial::f16>("x");
 auto cosine=spatial::input<1,64,spatial::f16>("cosine");
 auto sine=spatial::input<1,64,spatial::f16>("sine");
 #pragma csl dataflow rows=8 cols=8 partition=tiles coefficients=feature_pairs compute=dsd fp=relaxed
 auto rotated=spatial::rotate_pairs<spatial::pair_order::odd_even>(x,cosine,sine);
 spatial::output("rotated",rotated);
}
