def build_banner(title: str) -> str:
    bar = "=" * (len(title) + 4)
    return f"{bar}\n  {title}\n{bar}"


def stale_logger(message: str) -> None:
    # Used to be called from utils we have since deleted. dead-cst should flag this.
    print(f"[stale] {message}")
