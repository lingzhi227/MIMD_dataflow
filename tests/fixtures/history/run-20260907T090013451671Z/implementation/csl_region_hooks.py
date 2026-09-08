"""Explicit lowering-owned composition points in the shared CSL MLP engine."""

from frontend import check

SLOTS = (
    "ENTRY",
    "FINISH",
    "DECLARATIONS",
    "PARTIAL_RESET",
    "PARTIAL_MERGE",
    "ACCUM_RESET",
    "UP_FINISH",
    "GATE_FINISH",
)


def render(text, hooks=None):
    hooks = hooks or {}
    check(
        set(hooks) <= set(SLOTS) and all(isinstance(v, str) for v in hooks.values()),
        "known CSL region hooks",
    )
    for name in SLOTS:
        marker = "/*HLS_REGION_" + name + "*/"
        check(
            text.count(marker) == 1, "exactly one shared CSL composition point: " + name
        )
        text = text.replace(marker, hooks.get(name, ""))
    return text
