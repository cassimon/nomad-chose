"""One measurement file must produce exactly one NOMAD entry.

An upload from the Plains app carries, beside every measurement file, a
`<filename>.archive.yaml` describing it -- and that archive is the richer entry:
it links the sample and states the cell area and the illumination intensity,
none of which the instrument writes into the file. Parsing the raw file as well
would give the same measurement a second, poorer entry, and *both* would push a
solar cell into `archive.results` -- double-counting the device in the
perovskite-database overview plots.
"""

import shutil
from pathlib import Path

import pytest

from nomad_chose.parsers import parser_entry_point
from nomad_chose.parsers.parsers import ChoseParser

DATA = Path(__file__).parent.parent / 'data'

JV_FILE = '0001_2025-11-20_17.32.31_Stability (JV)_AI03-1A.txt'
PARAMETERS_FILE = '0000_2025-11-20_17.32.31_Stability (Parameters)_AI03-1A.txt'
TRACKING_FILE = '0000_2025-11-20_17.32.31_Stability (Tracking)_AI03-1A.txt'


@pytest.fixture
def parser():
    return ChoseParser(**parser_entry_point.model_dump())


def is_mainfile(parser, path: Path) -> bool:
    text = path.read_text(errors='replace')
    return bool(
        parser.is_mainfile(str(path), 'text/plain', text.encode(), text)
    )


def test_a_measurement_file_on_its_own_is_parsed(parser, tmp_path):
    """A file dropped into NOMAD by hand has no companion, and must still parse."""
    raw = tmp_path / JV_FILE
    shutil.copy(DATA / JV_FILE, raw)

    assert is_mainfile(parser, raw)


def test_a_measurement_file_the_app_already_described_is_skipped(parser, tmp_path):
    """The app's archive is the entry for this measurement; the raw file must not
    become a second one."""
    raw = tmp_path / JV_FILE
    shutil.copy(DATA / JV_FILE, raw)
    (tmp_path / f'{JV_FILE}.archive.yaml').write_text('data: {}\n')

    assert not is_mainfile(parser, raw)


def test_both_halves_of_a_stability_run_are_skipped(parser, tmp_path):
    """A stability run is two files but *one* measurement, so the app describes the
    pair in a single archive -- named after only one of them. The other half is
    spoken for just the same; parsing it would recreate the duplicate the companion
    archive exists to prevent.
    """
    params = tmp_path / PARAMETERS_FILE
    track = tmp_path / TRACKING_FILE
    shutil.copy(DATA / PARAMETERS_FILE, params)
    shutil.copy(DATA / TRACKING_FILE, track)

    # The app named its one archive after the (Parameters) half.
    (tmp_path / f'{PARAMETERS_FILE}.archive.yaml').write_text('data: {}\n')

    assert not is_mainfile(parser, params)
    assert not is_mainfile(parser, track)


def test_a_stability_pair_with_no_companion_is_still_parsed(parser, tmp_path):
    """Hand-dropped files: ChoseParser pairs them itself."""
    params = tmp_path / PARAMETERS_FILE
    track = tmp_path / TRACKING_FILE
    shutil.copy(DATA / PARAMETERS_FILE, params)
    shutil.copy(DATA / TRACKING_FILE, track)

    assert is_mainfile(parser, params)
    assert is_mainfile(parser, track)


def test_the_companion_is_matched_by_full_name_not_by_stem(parser, tmp_path):
    """The app names its archive after the whole raw filename, extension included.
    A same-stem archive is somebody else's file and must not suppress anything.
    """
    raw = tmp_path / JV_FILE
    shutil.copy(DATA / JV_FILE, raw)
    (tmp_path / f'{Path(JV_FILE).stem}.archive.yaml').write_text('data: {}\n')

    assert is_mainfile(parser, raw)
