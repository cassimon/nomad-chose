import pytest
from pathlib import Path

TEST_DATA = Path(__file__).parent / 'data'


@pytest.fixture
def jv_forward_csv():
    return str(TEST_DATA / 'jv_forward.csv')


@pytest.fixture
def jv_reverse_csv():
    return str(TEST_DATA / 'jv_reverse.csv')


@pytest.fixture
def jv_extra_csv():
    return str(TEST_DATA / 'jv_extra.csv')


@pytest.fixture
def not_a_jv_txt():
    return str(TEST_DATA / 'not_a_jv.txt')