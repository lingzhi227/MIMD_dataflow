"""Phase-aware inspection; child views rename witnesses, never execute transfers."""

from frontend import check
from score_softmax_debug import inspect as inspect_normalization
from device_matmul_debug import inspect as inspect_value


def inspect(s, results, node, epoch, step):
    p = s["P"]
    check(type(step) is int and 0 <= step <= 2 * p + 3, "attention debugger step")
    # A shallow diagnostic projection preserves the raw values and original epoch.
    view = None
    if results is not None:
        view = dict(results)
        if step <= p + 2:
            view["diagnostics"] = [
                dict(d, result=d["probability"]) for d in results.get("diagnostics", [])
            ]
        else:
            view["diagnostics"] = [
                dict(
                    d,
                    history=d["value_history"],
                    left_owners=d["value_left"],
                    right_owners=d["value_right"],
                    progress=d["value_progress"],
                )
                for d in results.get("diagnostics", [])
            ]
    if step <= p + 2:
        out = inspect_normalization(
            s["normalization_schedule"], view, node, epoch, step
        )
        # Probability is an internal observation in the composite counter profile.
        if s["instrumentation"] != "sampled" and step == p + 2:
            out.update(observed=False, half_bits=None)
        return out
    return inspect_value(s["value_schedule"], view, node, epoch, step - p - 3)
