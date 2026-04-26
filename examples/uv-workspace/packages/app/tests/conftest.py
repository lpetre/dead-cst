import pytest


@pytest.fixture
def runner():
    from app.cli import run

    return run
