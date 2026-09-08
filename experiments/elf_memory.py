"""Read linked resident ELF sections; config registers are outside PE SRAM.

This is static allocation evidence, not a runtime stack high-water measurement.
Use the pinned simtracer Python environment providing pyelftools on SDK host.
"""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, hashlib, json
from pathlib import Path
from elftools.elf.elffile import ELFFile

p = argparse.ArgumentParser()
p.add_argument("bundle", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
assert not a.output.exists(), "fresh evidence output required"
rows = []
for path in sorted((a.bundle / "out/bin").glob("*.elf")):
    with path.open("rb") as f:
        e = ELFFile(f)
        sections = [
            dict(name=s.name, start=int(s["sh_addr"]), size=int(s["sh_size"]))
            for s in e.iter_sections()
            if s["sh_flags"] & 2 and s["sh_addr"] < 49152
        ]
        high = max(s["start"] + s["size"] for s in sections)
        assert high <= 49152
        rows.append(
            dict(
                elf=path.name,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                sections=sections,
                static_high_water_bytes=high,
                unallocated_static_bytes=49152 - high,
            )
        )
assert rows
r = dict(
    scope="linked static SRAM sections by ELF class; not runtime stack usage",
    elf_classes=len(rows),
    max_static_high_water_bytes=max(x["static_high_water_bytes"] for x in rows),
    min_unallocated_static_bytes=min(x["unallocated_static_bytes"] for x in rows),
    classes=rows,
)
a.output.write_text(json.dumps(r, indent=2) + "\n")
print(
    r["elf_classes"],
    r["max_static_high_water_bytes"],
    r["min_unallocated_static_bytes"],
)
