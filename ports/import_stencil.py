"""Fail-closed extraction of scalar stencil.apply arithmetic from pinned MLIR.
The surrounding field/time loop is intentionally not imported; catalog says so.
"""

from pathlib import Path
import json, re, math

ROOT = Path(__file__).resolve().parent
REF = r"%[A-Za-z_0-9]+"


def extract(text):
    blocks = []
    active = False
    ops = []
    slots = []
    values = {}
    for line in text.splitlines():
        if "stencil.apply" in line:
            active = True
            ops = []
            slots = []
            values = {}
            continue
        if not active:
            continue
        if "stencil.return" in line:
            output = re.search(REF, line).group()
            blocks.append({"ops": ops, "slots": slots, "result": output})
            active = False
            continue
        match = re.match(r"\s*(" + REF + r")\s*=\s*(.*)", line)
        if not match:
            continue
        key, rhs = match.groups()
        args = re.findall(REF, rhs)
        if rhs.startswith("arith.constant"):
            raw = rhs.split()[1]
            value = float(raw)
            op = ["constant", key, value]
            values[key] = value
        elif "stencil.access" in rhs:
            offset = re.search(r"\[(-?\d+,\s*-?\d+,\s*-?\d+)\]", rhs)
            if not offset:
                raise ValueError("unsupported offset " + rhs)
            slot = {"field": args[0], "offset": [int(x) for x in offset[1].split(",")]}
            if slot not in slots:
                slots.append(slot)
            op = ["input", key, slots.index(slot)]
        elif rhs.startswith(("arith.addf", "arith.subf", "arith.mulf")):
            op = [rhs.split()[0].split(".")[1], key, *args]
        elif "math.fpowi" in rhs:
            if args[0] not in values or args[1] not in values:
                raise ValueError("only constant power folding supported")
            value = values[args[0]] ** int(values[args[1]])
            values[key] = value
            op = ["constant", key, value]
        elif rhs.startswith("arith.sitofp"):
            if args[0] not in values:
                raise ValueError("nonconstant cast")
            values[key] = values[args[0]]
            op = ["constant", key, values[key]]
        else:
            raise ValueError("unsupported MLIR arithmetic " + rhs)
        ops.append(op)
    return blocks


def emit(block):
    names = {}
    lines = ["spatial::tensor<1,1> out{};"]
    for i, op in enumerate(block["ops"]):
        kind, key, *args = op
        name = "s" + str(i)
        names[key] = name
        if kind == "constant":
            rhs = format(args[0], ".9e") + "f"
        elif kind == "input":
            rhs = "samples.data[" + str(args[0]) + "]"
        else:
            rhs = (
                names[args[0]]
                + {"addf": "+", "subf": "-", "mulf": "*"}[kind]
                + names[args[1]]
            )
        lines.append("float " + name + "=" + rhs + ";")
    lines += ["out.data[0]=" + names[block["result"]] + ";", "return out;"]
    return "\n".join(lines)


def reference(block, inputs):
    values = {}
    for kind, key, *args in block["ops"]:
        if kind == "constant":
            value = args[0]
        elif kind == "input":
            value = inputs[args[0]]
        else:
            a, b = values[args[0]], values[args[1]]
            value = {
                "addf": lambda: a + b,
                "subf": lambda: a - b,
                "mulf": lambda: a * b,
            }[kind]()
        values[key] = value
    return values[block["result"]]


def main():
    catalog = json.loads((ROOT / "catalog.json").read_text())
    for file in sorted((ROOT / "projects/wse_stencil/upstream").glob("*-small.mlir")):
        for i, block in enumerate(extract(file.read_text())):
            name = file.stem + "-apply" + str(i)
            d = ROOT / "projects/wse_stencil" / name
            d.mkdir(exist_ok=True)
            count = len(block["slots"])
            source = (
                '#include "spatial.hpp"\nvoid design(){auto samples=spatial::input<1,'
                + str(count)
                + '>("samples");auto result=spatial::kernel(samples,[](const spatial::tensor<1,'
                + str(count)
                + ">& samples){\n"
                + emit(block)
                + '\n});spatial::output("result",result);}\n'
            )
            (d / "hls.cpp").write_text(source)
            (d / "source-expression.json").write_text(
                json.dumps(block, indent=2) + "\n"
            )
            item = {
                "project": "wse_stencil",
                "kernel": name,
                "origins": [file.name],
                "fixture": "mlir:" + name,
                "contract": "Original stencil.apply scalar expression and offsets; constant integer powers folded. Surrounding distributed field and temporal loop remain unported.",
                "partitions": 1,
                "status": "source_ready",
            }
            (d / "PORT.json").write_text(json.dumps(item, indent=2) + "\n")
            catalog.append(item)
    (ROOT / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")


if __name__ == "__main__":
    main()
