# Public C++ HLS interfaces

Start with `spatial.hpp`. Additional headers define normalization, activation, pair rotation, blocked accumulation, FFT and resident solvers. Existing designs continue to include `"spatial.hpp"`; the compiler supplies this directory as an include root.

These headers express supported graph operations and native reference semantics. They do not replace CSL's execution or resource model. See [pragma syntax](../../docs/contracts/PRAGMAS.md).
