"""Check actual SDK ELF alignment/size for each directional staging buffer."""

import argparse, hashlib, json
from pathlib import Path
from elftools.elf.elffile import ELFFile
import ctf  # pinned simtracer ELF/LMA mapping reader

p = argparse.ArgumentParser()
p.add_argument("bundle", type=Path)
a = p.parse_args()
r = a.bundle.resolve()
s = json.loads((r / "schedule.json").read_text())
expected = 4 * ((s["capacity"]["rows"] + 1) // 2)
rows = []
for path in sorted((r / "out/bin").glob("*.elf")):
    with path.open("rb") as stream:
        table = ELFFile(stream).get_section_by_name(".symtab")
        found = {
            v.name: dict(address=int(v["st_value"]), bytes=int(v["st_size"]))
            for v in table.iter_symbols()
            if v.name in ("spmv_mod.west_row_wire", "spmv_mod.east_row_wire")
        }
        if not found:
            continue
        assert len(found) == 2, (path, found)
        for v in found.values():
            assert v["address"] % 4 == 0 and v["bytes"] == expected, (path, v)
        rows.append(
            dict(
                elf=path.name,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                buffers=found,
            )
        )
mapping = ctf.build_elf_mapping(
    [str(p) for p in (r / "out/bin").glob("*.elf")], s["cols"] + 7, quiet=True
)
by_name = {v["elf"]: v for v in rows}
actors = []
for y in range(s["rows"]):
    for x in range(s["cols"]):
        name = Path(mapping[(y + 1) * (s["cols"] + 7) + x + 4]).name
        assert name in by_name, (x, y, name)
        actors.append(dict(x=x, y=y, elf=name))
report = dict(
    passed=True,
    actors=len(actors),
    placements=actors,
    reader_sha256=hashlib.sha256(Path(ctf.__file__).read_bytes()).hexdigest(),
    bytes_per_direction=expected,
    scope="Actual application ELF symbol byte addresses and extents; checks 4-byte staging alignment, not remote-delivery completion",
    elfs=rows,
)
with (r / "staging-alignment-v2.json").open("x") as f:
    json.dump(report, f, indent=2)
print(json.dumps({k: v for k, v in report.items() if k not in ("elfs", "placements")}))
