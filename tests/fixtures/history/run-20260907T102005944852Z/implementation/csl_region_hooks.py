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
    "SETUP_BODY",
    "OBSERVE_BODY",
    "PHASE_FINISH_BODY",
    "PRESHIFT_RECORD",
    "STEP_RECORD",
)


DEFAULTS = {
    "PRESHIFT_RECORD": "if(phase==0){progress[0]=@as(u16,shift_round);}else{progress[1]=@as(u16,shift_round);}",
    "STEP_RECORD": "progress[2+phase]+=1;",
}


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
        text = text.replace(marker, hooks.get(name, DEFAULTS.get(name, "")))
    return text
