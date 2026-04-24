from simple_app.core import greet
from simple_app.utils import build_banner


def run() -> None:
    print(build_banner("simple-app"))
    print(greet("world"))


if __name__ == "__main__":
    run()
