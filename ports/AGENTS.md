# Port development

- Scope edits to this ports workspace. Preserve upstream reference trees and old sibling projects.
- Never overwrite a run directory or remove failed evidence. Fresh runs snapshot their implementation.
- Read PORT.json and upstream source before extending an application. Equation coverage is not full algorithm/schedule coverage.
- Keep numerical oracle and source-authoring scripts out of compiler/codegen. Extend shared typed IR rather than dispatching on kernel names.
- Run semantic regression tests and affected native/independent checks, then the SDK simulator. Record the actual level achieved.
- Do not update the remote toolchain during an active SDK batch: evidence validation checks its hashes.
- define_ports.py is an initial authoring script that resets the catalog; do not rerun it over accumulated ports. Other authoring scripts must be reviewed before execution.
- Regenerate status.py output after syncing preserved results; do not manually claim an unverified application passed.

## Direction confirmed by the user

- Work sequentially in this order: linear algebra, other numerical kernels, stencil, physical simulation, then other applications. Do not continue expanding stencil ahead of the linear-algebra queue.
- For each algorithm, express its high-performance dataflow in the HLS frontend, synthesize through the shared toolchain, run/debug the resulting CSL in SDK 2.10.1, and retain correctness plus scoped performance evidence. Develop the algorithm library and toolchain together. Mathematical equation coverage alone is not completion.

- This is an enhancement layer over CSL and SDK 2.10.1, not a replacement for them. Preserve access to CSL capabilities and reuse existing CSL libraries where appropriate.
- Prioritize faithful distributed numerical algorithms, realistic scale, and measured performance over adding more small equation-only examples.
- Learn target behavior through compiler probes, execution, intermediate-state inspection, failures, and comparison with SDK implementations. Do not infer resource or concurrency semantics only from syntax.
- Treat colors, queues, local tasks, microthreads, DSD/DSR ownership, halo exchange, Python SDK I/O, and PE memory as first-class constraints.
- Separate numerical equivalence, protocol correctness, compiler acceptance, simulator performance, and real-hardware performance evidence.
- Prepare clear, conventional architecture and source attribution for review by Cerebras HPC/CSL teams. Do not claim community endorsement. No public release has been authorized or performed in this task.
