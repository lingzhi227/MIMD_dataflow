"""Read the last instructions on each application PE from a saved matching CTF."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)

import argparse
import json
from pathlib import Path
from google.protobuf.json_format import MessageToDict
from cerebras.das_common.das_commonpybind import Coordinate
from cerebras.sdk.debug.lib.symbol.csldebugpybind import SimfabInstructionTraceCtf
p = argparse.ArgumentParser()
p.add_argument('root', type=Path)
p.add_argument('--cycle', type=int, required=True)
a = p.parse_args()
trace = SimfabInstructionTraceCtf(str(a.root/'out/bin'), str(a.root/'simfab_traces'))
result = {}
for y in range(1, 5):
    for x in range(4, 8):
        d = MessageToDict(trace.get_instruction_trace_at(Coordinate(x,y), 0, a.cycle+1), preserving_proto_field_name=True)
        rows = d.get('data', [])
        result[f'{x},{y}'] = rows[-12:]
(a.root/'instruction-tail.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
