# Runtime libraries

- [csl/](csl/): reusable CSL local math, collective adapters, routes, layouts and PE controllers.
- [native/](native/): C++ support used to execute HLS programs on the host for reference checks.
- Python SDK host bindings are separately located in [lib/Runtime/](../lib/Runtime/).

CSL module filenames and import relationships are retained to preserve generated-code behavior. Resource ownership, buffer lifetime and completion callbacks remain part of each library's contract. SDK binaries are external dependencies.
