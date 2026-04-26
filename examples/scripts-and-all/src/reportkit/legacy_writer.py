def write_legacy_report(path: str, body: str) -> None:
    # Module-wide dead code: nothing imports legacy_writer at all.
    with open(path, "w") as f:
        f.write(body)


def _legacy_header() -> str:
    return "=== legacy report ==="
