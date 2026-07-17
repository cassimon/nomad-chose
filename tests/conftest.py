import pytest
from pathlib import Path

TEST_DATA = Path(__file__).parent / 'data'



@pytest.fixture
def not_a_jv_txt():
    return str(TEST_DATA / 'not_a_jv.txt')


@pytest.fixture
def stability_jv_txt():
    return str(TEST_DATA / '0001_2025-11-20_17.32.31_Stability (JV)_AI03-1A.txt')


@pytest.fixture
def stability_parameters_txt():
    return str(TEST_DATA / '0000_2025-11-20_17.32.31_Stability (Parameters)_AI03-1A.txt')


@pytest.fixture
def stability_tracking_txt():
    return str(TEST_DATA / '0000_2025-11-20_17.32.31_Stability (Tracking)_AI03-1A.txt')


@pytest.fixture
def dark_jv_txt():
    return str(TEST_DATA / '2025-11-20_14.22.41_Dark JV_AI03.txt')


@pytest.fixture
def ipce_txt():
    return str(TEST_DATA / '2025-11-20_15.49.56_IPCE_AI03.txt')


@pytest.fixture
def uvvis_txt():
    return str(TEST_DATA / 'uvvis_transmittance.txt')