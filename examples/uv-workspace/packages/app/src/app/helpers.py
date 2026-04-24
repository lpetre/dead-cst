def legacy_helper(value: int) -> int:
    # Old helper, no longer referenced from cli or anywhere else in app.
    # dead-cst should flag this.
    return value - 1
