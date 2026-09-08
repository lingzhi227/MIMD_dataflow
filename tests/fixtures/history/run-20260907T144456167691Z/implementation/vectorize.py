"""Checked ordered affine-loop lowering to CSL vector operations.

Enumerates bounded integer indices, never numerical sample values. Requires the
same ordered coefficient/plane products at every lane. No reassociation or FMA.
"""

from frontend import check


def ordered_terms(kernel, z):
    env = {}
    writes = {}
    result = kernel["result"]
    check(
        set(kernel["arrays"]) == {"a", "b", result},
        "vectorize: unsupported scratch arrays",
    )

    def integer(e):
        if e[0] == "const" and e[2] == "int":
            return int(e[1])
        if e[0] == "var" and e[2] == "int":
            return env[e[1]]
        if e[0] == "binary" and e[4] in ("int", "bool"):
            a, b = integer(e[2]), integer(e[3])
            check(e[1] in ("+", "-", "*", "<"), "vectorize index expression")
            return {
                "+": lambda: a + b,
                "-": lambda: a - b,
                "*": lambda: a * b,
                "<": lambda: a < b,
            }[e[1]]()
        check(False, "vectorize integer expression")

    def floating(e):
        if e[0] == "var" and e[2] == "float":
            return list(env[e[1]])
        check(
            e[0] == "binary" and e[1] == "*" and e[4] == "float",
            "vectorize requires ordered coefficient products",
        )
        left, right = e[2:4]
        if left[:2] == ["index", "a"]:
            left, right = right, left
        check(
            left[:2] == ["index", "b"] and right[:2] == ["index", "a"],
            "vectorize coefficient/sample operands",
        )
        coeff, index = integer(left[2]), integer(right[2])
        check(
            0 <= coeff < kernel["arrays"]["b"][1] and 0 <= index < 7 * z,
            "vectorize access bounds",
        )
        return [(coeff, index)]

    def run(s):
        op = s[0]
        if op == "block":
            for child in s[1]:
                run(child)
        elif op == "array":
            check(s[1] == result and s[2] is None, "vectorize zero result storage")
        elif op == "decl":
            env[s[1]] = integer(s[3]) if s[2] == "int" else floating(s[3])
        elif op == "for":
            run(s[1])
            count = 0
            while integer(s[2]):
                check(count < 256, "vectorize loop bound")
                run(s[3])
                env[s[1][1]] += 1
                count += 1
        elif op == "assign":
            left, operator, right = s[1:]
            if left[0] == "var":
                check(
                    operator == "+=" and left[2] == "float",
                    "vectorize accumulator update",
                )
                env[left[1]] = env[left[1]] + floating(right)
            else:
                check(
                    left[:2] == ["index", result] and operator == "=",
                    "vectorize output assignment",
                )
                index = integer(left[2])
                check(index not in writes, "vectorize duplicate output write")
                writes[index] = floating(right)
        elif op == "return":
            check(s[1][:2] == ["var", result], "vectorize return")
        else:
            check(False, "vectorize unsupported control flow")

    run(kernel["body"])
    check(set(writes) == set(range(z)), "vectorize full output coverage")
    normalized = []
    for lane in range(z):
        terms = []
        for coeff, index in writes[lane]:
            check(
                (index - lane) % z == 0 and 0 <= index - lane < 7 * z,
                "vectorize nonuniform sample indexing",
            )
            terms.append([coeff, (index - lane) // z])
        normalized.append(terms)
    check(
        normalized[0] and all(v == normalized[0] for v in normalized),
        "vectorize nonuniform lanes",
    )
    return normalized[0]


def emit(terms, z):
    defs = [
        f"var vector_tmp=@zeros([{z}]f32);",
        f"const vector_out=@get_dsd(mem1d_dsd,.{{.tensor_access=|i|{{{z}}}->result[i]}});",
        f"const vector_temp=@get_dsd(mem1d_dsd,.{{.tensor_access=|i|{{{z}}}->vector_tmp[i]}});",
    ]
    for plane in sorted({t[1] for t in terms}):
        defs.append(
            f"const vector_in{plane}=@get_dsd(mem1d_dsd,.{{.tensor_access=|i|{{{z}}}->a[{plane*z}+i]}});"
        )
    code = []
    for i, (coeff, plane) in enumerate(terms):
        code.append(
            f'@fmuls({"vector_out" if i==0 else "vector_temp"},vector_in{plane},b[{coeff}]);'
        )
        if i:
            code.append("@fadds(vector_out,vector_out,vector_temp);")
    code.append(
        f"for(@range(u16,{z})) |i| {{@assert(result[i]==result[i] and math.abs(result[i])<3.4028234e38);}}"
    )
    return "\n".join(code), "\n".join(defs)
