"""Build honest per-port verification coverage from preserved evidence."""

import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    catalog = json.loads((ROOT / "catalog.json").read_text())
    inventory = json.loads((ROOT / "evidence/source_inventory.json").read_text())
    reports = []
    for p in sorted(
        list((ROOT / "evidence").glob("run-*.json"))
        + list((ROOT / "evidence").glob("qualification-*.json"))
    ):
        report = json.loads(p.read_text())
        for c in report.get("cases", []):
            reports.append((p, c))
    rows = []
    for item in catalog:
        key = item["project"] + "/" + item["kernel"]
        source = ROOT / "projects" / key / "hls.cpp"
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        matches = []
        for p, c in reports:
            if c["key"] != key or not c.get("passed"):
                continue
            artifact = ROOT / c["artifact"]
            manifest = artifact / "manifest.json"
            if not manifest.exists():
                continue
            m = json.loads(manifest.read_text())
            if m["source_sha256"] != digest:
                continue
            current = all(
                (ROOT / "toolchain" / name).exists()
                and hashlib.sha256((ROOT / "toolchain" / name).read_bytes()).hexdigest()
                == sha
                for name, sha in m["implementation"].items()
            )
            matches.append(
                dict(
                    level=c["level"],
                    artifact=c["artifact"],
                    report=str(p.relative_to(ROOT)),
                    native_application_source=c.get("native_application_source"),
                    current_toolchain=current,
                    numerical_validation=(
                        c.get("numerical_validation")
                        or (
                            {
                                "contract": c["audit"].get("acceptance_contract"),
                                "arithmetic_roundoff_passed": c["audit"].get(
                                    "arithmetic_roundoff_passed"
                                ),
                                "fixed_accuracy_passed": c["audit"].get(
                                    "fixed_accuracy_passed"
                                ),
                                "scope": "device per-round and final audit; CPU accuracy screen not recorded in this historical report",
                            }
                            if "arithmetic_roundoff_passed" in c.get("audit", {})
                            else None
                        )
                    ),
                )
            )
        rows.append(
            dict(
                key=key,
                contract=item["contract"],
                origins=item["origins"],
                source_sha256=digest,
                native_stdout_verification=max(
                    (v for v in matches if v.get("native_application_source")),
                    key=lambda v: v["report"],
                    default=None,
                ),
                verification=(
                    max(
                        matches,
                        key=lambda v: (v["level"] == "sdk_simulator", v["report"]),
                    )
                    if matches
                    else None
                ),
            )
        )
    result = dict(
        scope="Entries are bounded kernel profiles, NOT complete upstream applications. Referencing a source file does not mark that file fully ported.",
        native_evidence_scope="Historical native_application_checks fields may contain IR application checks. Direct C++ stdout evidence is separately identified; see docs/NATIVE-EVIDENCE.md.",
        ports=rows,
        reference_files=sum(len(p["files"]) for p in inventory.values()),
    )
    (ROOT / "evidence/status.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# 移植验证索引",
        "",
        "每一行代表一个有明确范围的移植 profile，不代表完整上游应用。具体限制见 contract 与 PORT.json。记录仅匹配当前 HLS 源码；工具链是否一致另列。",
        "",
        "真实 C++ stdout 应用检查在 evidence/status.json 的 native_stdout_verification 中单独记录；历史字段的范围更正见 [原生证据说明](docs/NATIVE-EVIDENCE.md)。SDK 证据不被新的 CPU 检查替代。",
        "",
        "| 内核 | 最新匹配验证 | 当前工具链 | 固定精度筛查 | 范围 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        v = r["verification"]
        link = f"[{v['level']}]({v['artifact']})" if v else "待验证"
        same = str(v["current_toolchain"]) if v else "—"
        numerical = v.get("numerical_validation") if v else None
        accuracy = (
            str(numerical.get("fixed_accuracy_passed"))
            if numerical
            else "未单列；见报告"
        )
        lines.append(
            f"| [{r['key']}](projects/{r['key']}/hls.cpp) | {link} | {same} | {accuracy} | {r['contract']} |"
        )
    (ROOT / "STATUS.md").write_text("\n".join(lines) + "\n")
    print(
        f'{len(rows)} bounded profiles; {result["reference_files"]} source reference files'
    )


if __name__ == "__main__":
    main()
