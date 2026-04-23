# tests/parsers/test_matching.py
"""
Tests that the ParserEntryPoint matching patterns behave correctly.
Uses the entry point's mainfile_name_re and mainfile_contents_re
directly — no NOMAD server needed.
"""

import re
import pytest
from nomad_chose.parsers import chose_jv_parser


class TestChoseJVParserMatching:

    @pytest.mark.parametrize('filename', [
        'measurement.jv.csv',
        'SAMPLE_JV_001.csv',
        'sample_jv_scan.csv',
        'data_JV_forward.csv',
        'run01.JV.csv',
    ])
    def test_matches_jv_filenames(self, filename):
        assert re.match(chose_jv_parser.mainfile_name_re, filename), (
            f'Expected {filename!r} to match but it did not.'
        )

    @pytest.mark.parametrize('filename', [
        'sample.archive.yaml',
        'README.txt',
        'data.csv',           # generic CSV, not JV
        'jv_notes.txt',
        'stability.csv',
    ])
    def test_does_not_match_non_jv_files(self, filename):
        assert not re.match(chose_jv_parser.mainfile_name_re, filename), (
            f'Expected {filename!r} NOT to match but it did.'
        )

    def test_content_check_matches_correct_header(self):
        content = '# operator: Alice\nvoltage,current_density\n0.0,21.3\n'
        assert re.search(chose_jv_parser.mainfile_contents_re, content)

    def test_content_check_rejects_wrong_header(self):
        content = 'time,temperature\n0.0,25.0\n'
        assert not re.search(chose_jv_parser.mainfile_contents_re, content)

    def test_entry_point_has_name(self):
        assert chose_jv_parser.name == 'ChoseJVParser'

    def test_entry_point_loads_parser(self):
        from nomad_chose.parsers.jv_parser import ChoseJVParser
        parser = chose_jv_parser.load()
        assert isinstance(parser, ChoseJVParser)