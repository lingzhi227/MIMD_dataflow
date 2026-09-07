"""Bounded raw CTF fallback; requires the separately pinned simtracer reader."""
import collections
import dataclasses
import hashlib
import json
import sys
from pathlib import Path
import ctf

root=Path(sys.argv[1])
selected={y*11+x for y in range(1,5) for x in range(4,8)}
tails={tile:collections.deque(maxlen=12) for tile in selected}
fabric={tile:collections.deque(maxlen=16) for tile in selected}
for stream in ctf.streams_for_tiles(str(root/'simfab_traces'),selected):
    for event in ctf.parse_ctf_stream(str(stream),want_ids=(2,),tile_filter=selected):
        tails[event.tile_index].append(dataclasses.asdict(event))
        if 'MOV' in event.name or 'LDS' in event.name:
            fabric[event.tile_index].append(dataclasses.asdict(event))
report=dict(reader=str(Path(ctf.__file__).resolve()),reader_sha256=hashlib.sha256(Path(ctf.__file__).read_bytes()).hexdigest(),root=str(root),scope='physical application rectangle 4..7,1..4, fabric width11; last12dispatch and last16MOV/LDS, not a full protocol proof',tails={k:list(v) for k,v in tails.items()},last_moves={k:list(v) for k,v in fabric.items()})
with (root/'raw-instruction-tail.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps({k:list(v)[-1] if v else None for k,v in tails.items()}))
