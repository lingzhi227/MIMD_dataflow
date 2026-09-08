"""Enumerate every normalized binary32 input for the pinned SDK magic seed.

This proves the initial seed interval used by the conservative refinement
analysis, not an exhaustive execution of SDK sqrt_f32 or a standalone ULP proof.
"""

import hashlib, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def run(output):
    output = Path(output)
    assert not output.exists()
    source = ROOT / "references/sdk-math-2.10.1/math.csl"
    text = source.read_text()
    assert "const MAGIC = @as(i32,0x5F3759DF);" in text
    lo, hi = 2.0, 0.0
    count = 0
    for start in range(0x3F800000, 0x40800000, 65536):
        words = np.arange(start, min(start + 65536, 0x40800000), dtype=np.uint32)
        x = words.view(np.float32).astype(float)
        y = (np.uint32(0x5F3759DF) - (words >> 1)).view(np.float32).astype(float)
        ratio = y * np.sqrt(x)
        lo = min(lo, float(ratio.min()))
        hi = max(hi, float(ratio.max()))
        count += len(words)
    assert 0.963 < lo <= hi < 1.037 and count == 2**24
    out = dict(
        passed=True,
        scope=__doc__,
        normalized_inputs=count,
        ratio_lower=lo,
        ratio_upper=hi,
        conservative_seed_interval=[0.963, 1.037],
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    output.write_text(json.dumps(out, indent=2) + "\n")
    return out


if __name__ == "__main__":
    print(run(sys.argv[1]))
