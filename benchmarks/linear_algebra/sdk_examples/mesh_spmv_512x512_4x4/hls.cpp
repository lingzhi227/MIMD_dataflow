#include "spatial.hpp"
void design() {
  auto values = spatial::input<4096,1>("values");
  auto row_indices = spatial::index_input<4096,1>("row_indices");
  auto column_offsets = spatial::index_input<513,1>("column_offsets");
  auto x = spatial::input<512,1>("x");
  #pragma csl dataflow rows=4 cols=4 storage=csc exchange=trains reduce=sparse_rows nnz_per_pe=512 cols_per_pe=128 rows_per_pe=128 fp=relaxed
  auto result = spatial::spmv_csc<512,512>(values,row_indices,column_offsets,x);
  spatial::output("result",result);
}
