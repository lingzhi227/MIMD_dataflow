#include "spatial.hpp"
void design() {
 auto x=spatial::input<128,1024,spatial::f16>("x");
 auto cosine=spatial::input<128,512,spatial::f16>("cosine");
 auto sine=spatial::input<128,512,spatial::f16>("sine");
 #pragma csl dataflow rows=8 cols=8 partition=tiles coefficients=per_token compute=dsd fp=relaxed
 auto rotated=spatial::rotate_pairs<spatial::pair_order::even_odd>(x,cosine,sine);
 spatial::output("rotated",rotated);
}
