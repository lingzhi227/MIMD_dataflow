"""Review pinned SDK DOT allocation diagnostics without confusing router colors."""

import argparse
import hashlib
import json
import re
from pathlib import Path


def parse(text, require_complete):
    nodes = {}
    for line in text.splitlines():
        match = re.match(r"\s*(\d+)\[.*?<b>([^<]+)</b>", line)
        if not match:
            continue
        node = int(match[1])
        assert node not in nodes
        assigned = re.search(r"Assigned color = (\d+)", line)
        explicit = re.search(r"Explicit DSR = (\d+)", line)
        allowed = re.search(r"Range = \[(\d+),(\d+)\)", line)
        assert explicit or allowed
        color = int(assigned[1]) if assigned else None
        if color is None:
            assert not require_complete and (
                "Failed before coloring this node" in line
                or "COLORING FAILED HERE" in line
            )
        elif explicit:
            assert color == int(explicit[1]), "Explicit DSR assignment changed"
        else:
            assert int(allowed[1]) <= color < int(allowed[2]), "DSR range violation"
        nodes[node] = dict(
            kind=match[2],
            assignment=color,
            explicit=int(explicit[1]) if explicit else None,
        )
    assert nodes
    edges = [tuple(map(int, v)) for v in re.findall(r"^\s*(\d+) -- (\d+)", text, re.M)]
    for a, b in edges:
        assert a in nodes and b in nodes
        x, y = nodes[a]["assignment"], nodes[b]["assignment"]
        if x is not None and y is not None:
            assert x != y, "Interfering DSR nodes share assignment"
    return dict(
        nodes=len(nodes),
        edges=len(edges),
        unassigned=sum(v["assignment"] is None for v in nodes.values()),
        explicit_ids=sorted(
            {v["explicit"] for v in nodes.values() if v["explicit"] is not None}
        ),
        automatic_ids=sorted(
            {
                v["assignment"]
                for v in nodes.values()
                if v["explicit"] is None and v["assignment"] is not None
            }
        ),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("probe", type=Path)
    p.add_argument("report", type=Path)
    a = p.parse_args()
    assert not a.report.exists()
    probe = a.probe.resolve()
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    execution = json.loads((probe / "report.json").read_text())
    provenance = json.loads((probe / "provenance.json").read_text())
    assert execution["compiler_success"] and execution["returncode"] == 0
    for name, digest in provenance["files"].items():
        assert sha(probe / name) == digest
    paths = sorted((probe / "out/bin").glob("*.dot"))
    assert len(paths) == 18
    reports = []
    for path in paths:
        complete = path.name.endswith(".dsatur.dot")
        reports.append(
            dict(
                file=str(path.relative_to(probe)),
                sha256=sha(path),
                phase=(
                    "complete allocation" if complete else "incomplete greedy attempt"
                ),
                **parse(path.read_text(), complete),
            )
        )
    # An edge conflict must be rejected; mutate an automatically assigned node
    # to its adjacent node's assignment in this diagnostic text only.
    text = next(p for p in paths if p.name.endswith(".dsatur.dot")).read_text()
    labels = {
        int(i): line
        for line in text.splitlines()
        if (m := re.match(r"\s*(\d+)\[", line))
        for i in [m[1]]
    }
    left, right = next(
        (int(x), int(y))
        for x, y in re.findall(r"^\s*(\d+) -- (\d+)", text, re.M)
        if "Range =" in labels[int(x)]
    )
    color = re.search(r"Assigned color = (\d+)", labels[right])[1]
    changed = re.sub(r"Assigned color = \d+", "Assigned color = " + color, labels[left])
    bad = text.replace(labels[left], changed)
    try:
        parse(bad, True)
    except AssertionError as error:
        negative = str(error)
    else:
        raise AssertionError("Invalid allocation accepted")
    a.report.write_text(
        json.dumps(
            dict(
                passed=True,
                graphs=reports,
                negative_check=negative,
                compiler_report_sha256=sha(probe / "report.json"),
                provenance_sha256=sha(probe / "provenance.json"),
                driver_sha256=sha(Path(__file__)),
                scope="Pinned SDK compiler diagnostic consistency: graph assignments, explicit DSR IDs and interference edges. Greedy attempt is incomplete while dsatur graphs are complete. Graph colors denote register assignments, not fabric routing colors. No dynamic stack or global task-order proof.",
            ),
            indent=2,
        )
        + "\n"
    )
    print("DSR DIAGNOSTICS PASS", len(reports), flush=True)


if __name__ == "__main__":
    main()
