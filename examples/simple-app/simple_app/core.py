def greet(name: str) -> str:
    return _format_message("Hello", name)


def _format_message(prefix: str, name: str) -> str:
    return f"{prefix}, {name}!"


def legacy_greet(name: str) -> str:
    # Older API, no longer called from anywhere. dead-cst should flag this.
    return "Hi, " + name
