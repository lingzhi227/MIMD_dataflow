# HLS reading tutorials

These guided paths point to existing, tested sources rather than duplicate them. This follows the distinction between learning examples and algorithm benchmarks used by Cerebras, and the feature-oriented navigation used by AMD's HLS examples.

1. **Tensor operators and source contracts:** start with `sdk_examples/gemm` in the [profile index](../../validation/STATUS.md), then read [the public interface](../../include/pragma/README.md).
2. **Spatial mapping and communication:** read [distributed GEMV](../../docs/contracts/MESH-GEMV.md), [SUMMA](../../docs/contracts/SUMMA.md) and [pragma syntax](../../docs/contracts/PRAGMAS.md).
3. **Precision policy:** read [blocked MLP](../../docs/contracts/RECTANGULAR-MLP.md) and [mean-statistic RMS](../../docs/contracts/MEAN-STATISTIC-RMS.md).
4. **Resident composition:** use the [example tour](../README.md), progressing from supplied Q/K/V attention to the qualified projected-cache graph. The larger attention-plus-FFN parent remains pending qualification.
5. **Inspect the generated program:** use `tools/hls_debug.py` on a fresh run. Recorded results are not a substitute for all-stage mathematical and protocol checks.
