import pytest


@pytest.fixture
def doubled():
    from core.api import used_by_app

    return used_by_app
