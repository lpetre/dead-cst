def format_csv(rows: list[dict]) -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())
    lines = [",".join(headers)]
    for row in rows:
        lines.append(",".join(str(row[h]) for h in headers))
    return "\n".join(lines)


def format_xml(rows: list[dict]) -> str:
    # Never wired up to render() and not re-exported via __all__.
    # dead-cst should flag this.
    parts = ["<rows>"]
    for row in rows:
        parts.append("  <row>")
        for k, v in row.items():
            parts.append(f"    <{k}>{v}</{k}>")
        parts.append("  </row>")
    parts.append("</rows>")
    return "\n".join(parts)
