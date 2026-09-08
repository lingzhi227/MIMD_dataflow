#include "spatial.hpp"
void design() {
  auto values = spatial::input<32768,1>("values");
  auto row_indices = spatial::index_input<32768,1>("row_indices");
  auto column_offsets = spatial::index_input<4097,1>("column_offsets");
  auto x = spatial::input<4096,1>("x");
  #pragma csl dataflow rows=8 cols=8 storage=csc exchange=trains reduce=sparse_rows nnz_per_pe=1024 cols_per_pe=512 rows_per_pe=512 fp=relaxed
  auto result = spatial::spmv_csc<4096,4096>(values,row_indices,column_offsets,x);
  spatial::output("result",result);
}
