# Supplied-cache attention with output residual

Qualified bounded contract: B=5, N=256, S=512 on an 8×8 PE region. This port follows the device contraction directions in WaferLLM Decode. It consumes already-rotated queries/keys and a shared read-only cache; cache mutation, causal masking and multihead/GQA selection are outside this boundary.

At PE(y,x), K is the feature-major local view `K[y*St+j,x*Nt+i]`, stored at `i*St+j`. V uses the same logical block stored at `j*Nt+i`. Queries partition features along X; score/probability partition sequence along Y; context partitions features along X; output and residual partition features along Y. Non-square cache and directed coordinate fixtures must validate these directions in actual SDK execution. The restored upstream Python K packing is not an oracle.

Half local DSR contractions use SDK f32 reduce/broadcast with a final half narrow. Row maximum uses gather plus broadcast through the same SDK instances. Softmax uses true negative maxima, SDK half exp, stationary local summation and half normalization. No intermediate host arithmetic is permitted.

Qualified as configuration 139: eight actual SDK calls, eight original local-compute control calls, two-host native checks, five independent numerical stage gates and 67 rejected corruptions. All 20 observed groups match the compute control exactly. Simulator maximum PE cycles are 52,157 versus 52,975; this is a local-lowering comparison, not hardware or full Decode throughput.

The frontend now explicitly uses 32-element half FMA blocks and float merging. PV splits its 64 resident sequence positions into two blocks. The original coherent and cancellation fixtures remain unchanged. See [the full contract and evidence](../../../docs/SUPPLIED-CACHE-ATTENTION.md) for layers, resources, preserved failures and current qualification status.
