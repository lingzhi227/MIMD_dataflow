"""Explicit frontend precision for the experimentally executed resident algorithm.

No qualified mixed-profile dispatch yet; unsupported policies must reject.
"""

from input_attention_source import source as half_source


def source(m=64, n=64, f=256, p=8):
    text = half_source(m, n, f, p)
    for a, b in (
        ("input_normalized", "v_weight"),
        ("probability", "v"),
        ("attention", "output_weight"),
    ):
        old = f"spatial::matmul({a},{b})"
        assert text.count(old) == 1
        text = text.replace(
            old, f"spatial::matmul<spatial::scalar,spatial::scalar>({a},{b})"
        )
    text = text.replace(
        "spatial::softmax(score,", "spatial::softmax<spatial::scalar>(score,"
    )
    text = text.replace(
        "spatial::add(projection,input_x)",
        "spatial::add<spatial::scalar>(projection,input_x)",
    )
    text = text.replace(
        "spatial::rmsnorm(z,gamma,",
        "spatial::rmsnorm<spatial::f16,spatial::scalar>(z,gamma,",
    )
    text = text.replace("spatial::add(z,delta)", "spatial::add<spatial::f16>(z,delta)")
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "spatial::matmul<spatial::scalar" in line:
            lines[i - 1] += " accumulation=f32"
        elif (
            "spatial::softmax<spatial::scalar" in line
            or "spatial::rmsnorm<spatial::f16,spatial::scalar>" in line
        ):
            lines[i - 1] = (
                lines[i - 1]
                .replace("accumulation=f16", "accumulation=f32")
                .replace("math=sdk_half", "math=sdk_float")
            )
    return "\n".join(lines) + "\n"
