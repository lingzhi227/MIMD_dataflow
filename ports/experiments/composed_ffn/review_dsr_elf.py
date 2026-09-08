"""Check whether the diagnostic flag preserved linked resident sections."""

import argparse
import hashlib
import json
from pathlib import Path
from elftools.elf.elffile import ELFFile

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def sections(path):
    with path.open("rb") as f:
        elf = ELFFile(f)
        return {
            s.name: dict(
                start=int(s["sh_addr"]),
                size=int(s["sh_size"]),
                sha256=hashlib.sha256(s.data()).hexdigest(),
            )
            for s in elf.iter_sections()
            if s["sh_flags"] & 2 and s["sh_addr"] < 49152
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bundle", type=Path)
    p.add_argument("probe", type=Path)
    p.add_argument("report", type=Path)
    a = p.parse_args()
    assert not a.report.exists()
    rows = []
    for x in sorted((a.bundle / "out/bin").glob("*.elf")):
        y = a.probe / "out/bin" / x.name
        before, after = sections(x), sections(y)
        rows.append(
            dict(
                elf=x.name,
                baseline_sha256=sha(x),
                diagnostic_sha256=sha(y),
                resident_sections_identical=before == after,
                before=before,
                after=after,
            )
        )
    assert len(rows) == 9
    report = dict(
        passed=all(v["resident_sections_identical"] for v in rows),
        classes=rows,
        driver_sha256=sha(Path(__file__)),
        scope="Compare all allocatable SRAM section locations, sizes and bytes between standard build and SDK DSR-dump build; debug sections excluded. No additional simulator execution.",
    )
    a.report.write_text(json.dumps(report, indent=2) + "\n")
    print("RESIDENT SECTIONS IDENTICAL", report["passed"], flush=True)


if __name__ == "__main__":
    main()
