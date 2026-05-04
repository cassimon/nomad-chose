import re

import pytest

from nomad_chose.parsers import parser_entry_point


class TestUnifiedParserMatching:
    @pytest.mark.parametrize(
        'filename',
        [
            'measurement.jv.csv',
            '0001_2025-11-20_17.32.31_Stability (JV)_AI03-1A.txt',
            '0000_2025-11-20_17.32.31_Stability (Parameters)_AI03-1A.txt',
            '0000_2025-11-20_17.32.31_Stability (Tracking)_AI03-1A.txt',
            '2025-11-20_15.49.56_IPCE_AI03.txt',
        ],
    )
    def test_name_regex_matches_supported_extensions(self, filename):
        assert re.match(parser_entry_point.mainfile_name_re, filename)

    @pytest.mark.parametrize(
        'filename',
        [
            'sample.archive.yaml',
            'notes.md',
            'report.pdf',
            'image.png',
        ],
    )
    def test_name_regex_rejects_unsupported_extensions(self, filename):
        assert not re.match(parser_entry_point.mainfile_name_re, filename)

    def test_txt_guard_accepts_true_stability_content(self):
        content = 'Header\nTest\tStability (JV)\n## Data ##\n'
        assert re.search(parser_entry_point.mainfile_contents_re, content)

    def test_txt_guard_rejects_unrelated_txt_content(self):
        content = 'Header\nTest\tRandom Note\n## Data ##\n'
        assert not re.search(parser_entry_point.mainfile_contents_re, content)

    def test_entry_point_loads_unified_parser(self):
        parser = parser_entry_point.load()
        assert parser.__class__.__name__ == 'ChoseParser'
