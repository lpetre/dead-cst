from reportkit.formatters import format_csv

__all__ = ["render"]


def render(rows: list[dict], *, fmt: str = "csv") -> str:
    if fmt == "csv":
        return format_csv(rows)
    raise ValueError(f"unsupported format: {fmt}")
