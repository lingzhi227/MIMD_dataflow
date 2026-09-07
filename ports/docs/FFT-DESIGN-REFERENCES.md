# FFT interface and dataflow references

The executable foundation is the actual SDK2.10.1 FFT/transpose library, with preserved source hashes under `references/sdk-fft-2.10.1`. The HLS frontend chooses numerical semantics and output ownership; it does not implement an FPGA circuit backend.

[FFTW's transposed-distribution interface](https://www.fftw.org/fftw3_doc/Transposed-distributions.html) explicitly permits transposed output to omit a final redistribution. This supports offering output ownership as a performance choice. Our two-dimensional PE-pencil distribution and physical axes differ from FFTW's MPI distribution; the APIs and layouts are not claimed compatible.

The already downloaded AMD reference manifest (archive reference: `../../references/amd/manifest.json`) pins Vitis_Libraries to3ab1ecd20cf338b3f8f2824fdfeee115f3398889 and Introductory-Examples toaa5c160faf5d5ebf58674df8f0591f9984ebae0f. In the local `dsp/L1/include/hw/vitis_fft/float/vitis_fft/hls_ssr_fft_enums.hpp`, output order is a configuration field distinct from transform direction. Its natural/digit-reversed-transposed choice is a design reference, not the same as our distributed output permutation. No AMD implementation or pipeline-II guarantee is incorporated into this CSL lowering.

[Wafer-Scale Fast Fourier Transforms](https://arxiv.org/abs/2209.15040) describes local pencils and mesh-axis redistributions. This is relevant to the family of mapping used here; our SDK simulator cases do not reproduce its CS-2 hardware measurements. The [author's publication page](https://morenes.github.io/) links the paper but did not provide a standalone source repository for it in this review. Do not label current SDK results a full paper reproduction without resolving the exact source/version relationship.

[Slide FFT](https://arxiv.org/abs/2401.05427) describes another approach based on adjacent-PE sliding. It is distinct from the SDK pencil-transpose algorithm. The reported sliding-window repository lookup returned not found during this review; no source port or execution is claimed for that work.

## Explicit output contract

`result=input_layout` keeps the current SDK behavior, including two final transposes that restore input ownership. `result=transposed_pencils` stops after the third local FFT, with physical axes `[x,z,y]`; host gathering only permutes layout back to logical `[y,x,z]`, without computing an FFT. A future resident consumer could use that declared physical ownership directly. Transposed-input/resident forward-inverse composition is not implemented.

The optional controller adapter changes only observer callbacks and early completion; FFT arithmetic and both required transposes remain SDK modules. The original SDK resets routes and teardown bookkeeping at each transpose_start; next_vertical_mode is assigned but unread in this pinned implementation. Warm re-entry must be tested rather than inferred. See `evidence/fft-driver-source-diff.txt`. The32³/8x8 transposed-output bundle195708248348 now passes six warm SDK calls, all five sampled stages, and exact logical device-bit comparison with restored output. Matched sampled max-local cycles are202532versus224046; no hardware or host-gather speedup is claimed.

> Packaging note: archive references identify original experimental artifacts or research references not bundled in this curated checkout. They are not local download links. See the root release selection policy.
