"""Same-input cycle comparison; simulator cycles, not wall clock or hardware FLOPS."""

import argparse, json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("vector")
p.add_argument("scalar")
p.add_argument("-o", required=True)
a = p.parse_args()
v = Path(a.vector)
s = Path(a.scalar)


def read(root, name):
    return json.loads((root / name).read_text())


assert read(v, "batches.json") == read(s, "batches.json"), "different inputs"
vs, ss = read(v, "schedule.json"), read(s, "schedule.json")
for key in ("grid", "epochs", "wire_z", "coefficients", "host_input_size"):
    assert vs[key] == ss[key], key
assert vs["vector_terms"] and ss["vector_terms"] is None
vr, sr = read(v, "results.json"), read(s, "results.json")
assert (
    vr["success"] and sr["success"] and vr["cases"] == sr["cases"]
), "final outputs not identical"
assert all(
    vr["diagnostics"][name]["history"] == sr["diagnostics"][name]["history"]
    for name in vr["diagnostics"]
), "timestep bits differ"
vc, sc = (
    read(v, "sim_stats.json")["cycle_count"],
    read(s, "sim_stats.json")["cycle_count"],
)
result = {
    "vector_artifact": str(v),
    "scalar_artifact": str(s),
    "vector_cycles": vc,
    "scalar_cycles": sc,
    "scalar_over_vector": sc / vc,
    "same_inputs": True,
    "all_step_float_bits_identical": True,
    "scope": "Total SDK simulator cycles including host IO and full history instrumentation. Same DSD communication, different compute lowering. Not a comparison with hand-written SDK stencil or a hardware performance result.",
}
Path(a.o).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
