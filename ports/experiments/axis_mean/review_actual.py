"""Independent original-input mathematical gate using actual SDK output words."""

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path


def decode(word):
    return struct.unpack("<e", struct.pack("<H", word))[0]


def review(root):
    data = json.loads((root / "inputs.json").read_text())
    result = json.loads((root / "results.json").read_text())
    schema = json.loads((root / "schema.json").read_text())
    p = schema["rows"]
    nt = schema["inputs"]["Z"] // 3
    assert p == schema["cols"] and p * nt == 256
    checks = []
    assert result["success"] and len(result["cases"]) == len(data) == 8
    divisors = (256, 256, 1, 256, 128, 256, 512, 256)
    for epoch, (batch, case, divisor) in enumerate(
        zip(data, result["cases"], divisors)
    ):
        errors, ideal_values = [], []
        for y in range(p):
            for x in range(p):
                for b in range(3):
                    coords = (
                        [(y, i) for i in range(p)]
                        if epoch % 2 == 0
                        else [(i, x) for i in range(p)]
                    )
                    total = math.fsum(
                        batch["Z"][yy][xx][b * nt + j] ** 2
                        for yy, xx in coords
                        for j in range(nt)
                    )
                    inverse = 1 / math.sqrt(total / divisor + 1e-6)
                    for j in range(nt):
                        ideal = batch["Z"][y][x][b * nt + j] * inverse
                        actual = decode(case["normalized"][y][x][b * nt + j])
                        assert math.isfinite(actual)
                        errors.append(actual - ideal)
                        ideal_values.append(ideal)
        norm = math.sqrt(math.fsum(v * v for v in ideal_values))
        peak = max(map(abs, ideal_values))
        l2 = math.sqrt(math.fsum(v * v for v in errors)) / (norm or 1)
        relpeak = max(map(abs, errors)) / (peak or 1)
        checks.append(
            dict(
                epoch=epoch,
                divisor=divisor,
                values=len(errors),
                relative_l2=l2,
                relative_peak=relpeak,
            )
        )
        assert l2 <= 0.02 and relpeak <= 0.03
        if epoch < 2:
            assert all(
                math.isinf(decode(case["sum_box"][y][x][b + 1]))
                for y in range(8)
                for x in range(8)
                for b in range(3)
            )
            assert all(
                math.isfinite(decode(case["mean_box"][y][x][b + 1]))
                for y in range(8)
                for x in range(8)
                for b in range(3)
            )
    files = [
        root / n
        for n in (
            "inputs.json",
            "results.json",
            "provenance.json",
            "execution.json",
            "schema.json",
        )
    ] + [Path(__file__)]
    return dict(
        passed=True,
        checks=checks,
        values_checked=sum(c["values"] for c in checks),
        files={
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
        scope="Standard-library original-input equation and actual half word decoding; no target arithmetic predictor. Eight RMS/generalized cases; no end-to-end HLS or throughput claim.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("report", type=Path)
    a = p.parse_args()
    assert not a.report.exists()
    r = review(a.root)
    a.report.write_text(json.dumps(r, indent=2) + "\n")
    print("PASS", r["values_checked"], max(c["relative_l2"] for c in r["checks"]))
