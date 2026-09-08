"""Public compiler profile for the checked resident attention/FFN composition."""

import copy
from frontend import check
from projected_cache_ffn_ir import canonical
from projected_cache_ffn_plan import plan as parent_plan
from projected_cache_ffn_codegen import generate

PROFILE = "mesh_projected_cache_ffn.v1"


def verify(module, epochs, bound):
    graph = canonical(module)
    a = graph["attention"]["nodes"]
    t = graph["ffn"]["nodes"]
    m = copy.deepcopy(module)
    m["nodes"] = a[:10] + t[2:5] + a[10:22] + t[5:12] + [t[-1], a[23], a[24]]
    for n in m["nodes"]:
        n["interval"] = None
        if n["op"] == "output":
            n.update(dtype="f16", shape=a[0]["shape"][:])
    m.update(profile=PROFILE, epochs=epochs, input_bound=bound)
    plan(m)
    return m


def plan(m, partitions=1):
    check(partitions == 1, "composed graph single resident region")
    s = parent_plan(m, m["epochs"], m["input_bound"])
    a = s["attention"]
    for k in (
        "P",
        "B",
        "N",
        "S",
        "Nt",
        "St",
        "rows",
        "cols",
        "epochs",
        "nodes",
        "instrumentation",
    ):
        s[k] = a[k]
    s.update(
        profile=PROFILE,
        ownership=dict(
            a["ownership"],
            ffn="Z feature Y -> UP/GATE feature X -> DOWN feature Y; original Z residual; source-shared gamma",
        ),
    )
    return s


def evaluate(m, batches):
    from projected_cache_ffn_reference import native_graph

    check(len(batches) == m["epochs"], "composed native epochs")
    rows = []
    for batch in batches:
        values = native_graph(m, batch)
        rows.append(
            {
                n["host"]: values[n["inputs"][0]].ravel().tolist()
                for n in m["nodes"]
                if n["op"] == "output"
            }
        )
    return rows, {}
