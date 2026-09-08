# SDK 2.10.1 conditional subscript compiler diagnostic

The pinned SDK image and compiler are unchanged. A generated MLP first failed parsing because `layout` was used as a variable name (fixed). Subsequent preserved builds failed inside `AstToCsl.cpp:162`, `toPointerType`, before execution.

The isolated trigger is:

```csl
var values=@zeros([2]u16);
var phase:i16=0;
fn run() void {
  values[if(phase==0) 0 else 1]=1;
  phase+=1;
}
```

Both following forms compile under the same SDK:

```csl
const index:i16=if(phase==0) 0 else 1;
values[index]=1;
```

```csl
if(phase==0){values[0]=1;}else{values[1]=1;}
```

Evidence: `evidence/conditional-subscript-20260907T044450072235Z`, all inputs and logs hashed. The inline form returns250 with an internal assertion; typed/branch forms return0. This is a compile-only diagnostic; it does not establish execution equivalence of the tiny programs. Full generated MLP pair `mlp-compile-isolation-20260907T044301516176Z` changes only this statement and similarly switches return250→0. All-function stubbing/restoration `044148090349` independently isolated the preshift callback. The HLS runtime uses the explicit branch form.

Earlier pointer-conditional and DSD pointer-cast hypotheses did not fix the full program. Minimal array-pointer and many-pointer DSD controls both compile (`dsd-pointer-type-20260907T044049370820Z`). The first such probe omitted required channels and failed in the driver, so it is preserved as an inconclusive harness failure, not compiler evidence. No claim about all conditional subscripts or other SDK versions follows from this case.

Reproduction uses the frozen driver in each fresh probe, not modifications to existing evidence. The report `evidence/conditional-subscript-sdk2101-review.json` links all provenance and output hashes. No external bug report has been sent.
