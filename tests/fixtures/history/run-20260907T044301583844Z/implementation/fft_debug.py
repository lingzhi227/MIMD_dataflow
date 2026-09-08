"""Inspect restored FFT pencils without claiming access to private SDK stages."""

import re
from frontend import check


def inspect(s, r, node, epoch, step):
    match = re.fullmatch(r"p(\d+)_(\d+)", node)
    check(match is not None, "FFT PE p<column>_<row>")
    x, y = map(int, match.groups())
    p = s["rows"]
    t = s["T"]
    n = s["N"]
    check(
        0 <= x < p and 0 <= y < p and 0 <= epoch < s["epochs"] and 0 <= step < t * t,
        "FFT PE/epoch/local pencil bounds",
    )
    out = dict(
        node=node,
        epoch=epoch,
        local_pencil=step,
        logical_y=y * t + step // t,
        logical_x=x * t + step % t,
        internal_phases_observed=s.get("internal_phases_observed", False),
        observed=False,
    )
    if r and epoch < len(r.get("diagnostics", [])):
        d = r["diagnostics"][epoch]
        v = d["packed_output"][y][x]
        if d.get("phase_samples") is not None:
            values = d["phase_samples"][y][x]
            out["phase_endpoint_order"] = [
                "first_real",
                "first_imag",
                "last_real",
                "last_imag",
            ]
            out["phase_endpoints"] = [
                values[phase * 4 * t * t + 4 * step : phase * 4 * t * t + 4 * step + 4]
                for phase in range(s.get("phase_count", 7))
            ]
            out["phase_counts"] = d["phase_counts"][y][x]
        out.update(
            observed=True,
            final_complex_pencil=[
                v[2 * (z * t * t + step) : 2 * (z * t * t + step) + 2] for z in range(n)
            ],
            progress=d["progress"][y][x],
            queue_masks=d["queues"][y][x],
            elapsed_timestamp_limbs=d["timing"][y][x],
        )
    out["varying_axis"] = "z"
    if s.get("result_layout") == "transposed_pencils":
        out["logical_x"] = y * t + step // t
        out["logical_z"] = x * t + step % t
        out["logical_y"] = None
        out["varying_axis"] = "y"
    return out
