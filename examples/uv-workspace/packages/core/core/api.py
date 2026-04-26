def used_by_app(value: int) -> int:
    return value * 2


def unused_old(value: int) -> int:
    # Internal-only API, no remaining callers in app/. dead-cst should flag this.
    return value + 1
