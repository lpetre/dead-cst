from reportkit.renderer import render


def main() -> None:
    rows = [{"name": "Alice", "score": 42}, {"name": "Bob", "score": 17}]
    print(render(rows, fmt="csv"))
