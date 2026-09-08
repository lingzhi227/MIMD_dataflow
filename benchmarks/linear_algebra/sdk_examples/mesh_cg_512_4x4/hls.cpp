#include "spatial.hpp"
void design() {
  auto values = spatial::input<4608,1>("values");
  auto rows = spatial::index_input<4608,1>("row_indices");
  auto columns = spatial::index_input<513,1>("column_offsets");
  auto rhs = spatial::input<512,1>("rhs");
  auto initial = spatial::input<512,1>("initial");
  auto limit = spatial::index_input<1,1>("iteration_limit");
  auto tolerances = spatial::input<2,1>("tolerances");
  #pragma csl dataflow rows=4 cols=4 storage=csc exchange=trains reduce=row_column redistribute=transpose recurrence=resident compute=vector nnz_per_pe=512 cols_per_pe=128 rows_per_pe=128 fp=relaxed
  auto solved = spatial::cg_csc<512,64>(values,rows,columns,rhs,initial,limit,tolerances);
  spatial::output("solution",solved.solution);
  spatial::output("reason",solved.reason);
  spatial::output("iterations",solved.iterations);
  spatial::output("residual_squared",solved.residual_squared);
  spatial::output("true_residual_norm",solved.true_residual_norm);
}
