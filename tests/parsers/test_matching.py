# tests/parsers/test_matching.py
"""
Tests that the ParserEntryPoint matching patterns behave correctly.
Uses the entry point's mainfile_name_re and mainfile_contents_re
directly — no NOMAD server needed.
"""
import re
import pytest


# ── The patterns under test — copied from parsers/__init__.py ────────────────
# Testing them here as literals means this file has zero imports from
# nomad_chose and cannot trigger any circular import.

JV_NAME_RE       = r'.*\.(jv\.csv|JV\.csv)|.*_[Jj][Vv]_.*\.csv'
JV_CONTENTS_RE   = r'voltage,current_density'
EQE_NAME_RE      = r'.*\.(eqe\.csv|EQE\.csv)|.*_[Ee][Qq][Ee]_.*\.csv'
EQE_CONTENTS_RE  = r'wavelength,eqe'


class TestJVNamePattern:

    @pytest.mark.parametrize('filename', [
        'measurement.jv.csv',
        'measurement.JV.csv',
        'SAMPLE_JV_001.csv',
        'sample_jv_scan.csv',
        'data_JV_forward.csv',
        'run01.JV.csv',
    ])
    def test_matches(self, filename):
        assert re.match(JV_NAME_RE, filename), (
            f'{filename!r} should match JV pattern.'
        )

    @pytest.mark.parametrize('filename', [
        'sample.archive.yaml',
        'README.txt',
        'data.csv',
        'jv_notes.txt',
        'stability.csv',
        'run01.eqe.csv',    # EQE should not match JV
    ])
    def test_does_not_match(self, filename):
        assert not re.match(JV_NAME_RE, filename), (
            f'{filename!r} should NOT match JV pattern.'
        )


class TestJVContentsPattern:

    def test_matches_correct_header(self):
        assert re.search(JV_CONTENTS_RE,
                         '# operator: Alice\nvoltage,current_density\n')

    def test_rejects_eqe_header(self):
        assert not re.search(JV_CONTENTS_RE, 'wavelength,eqe\n400,0.6\n')

    def test_rejects_generic_csv(self):
        assert not re.search(JV_CONTENTS_RE, 'time,temperature\n0,25\n')


# class TestEQENamePattern:

#     @pytest.mark.parametrize('filename', [
#         'measurement.eqe.csv',
#         'measurement.EQE.csv',
#         'SAMPLE_EQE_001.csv',
#         'sample_eqe_scan.csv',
#         'data_EQE_run1.csv',
#         'run01.EQE.csv',
#     ])
#     def test_matches(self, filename):
#         assert re.match(EQE_NAME_RE, filename), (
#             f'{filename!r} should match EQE pattern.'
#         )

#     @pytest.mark.parametrize('filename', [
#         'sample.archive.yaml',
#         'data.csv',
#         'run01.jv.csv',     # JV should not match EQE
#     ])
#     def test_does_not_match(self, filename):
#         assert not re.match(EQE_NAME_RE, filename), (
#             f'{filename!r} should NOT match EQE pattern.'
#         )


# class TestEQEContentsPattern:

#     def test_matches_correct_header(self):
#         assert re.search(EQE_CONTENTS_RE,
#                          '# operator: Bob\nwavelength,eqe\n400,0.6\n')

#     def test_rejects_jv_header(self):
#         assert not re.search(EQE_CONTENTS_RE,
#                              'voltage,current_density\n0.0,21.3\n')


# class TestPatternsAreDisjoint:
#     """JV and EQE name patterns must never match the same filename."""

#     @pytest.mark.parametrize('jv_file', [
#         'run1.jv.csv', 'data_JV_001.csv',
#     ])
#     def test_jv_file_does_not_match_eqe(self, jv_file):
#         assert     re.match(JV_NAME_RE,  jv_file)
#         assert not re.match(EQE_NAME_RE, jv_file)

#     @pytest.mark.parametrize('eqe_file', [
#         'run1.eqe.csv', 'data_EQE_001.csv',
#     ])
#     def test_eqe_file_does_not_match_jv(self, eqe_file):
#         assert     re.match(EQE_NAME_RE, eqe_file)
#         assert not re.match(JV_NAME_RE,  eqe_file)


# ── Optional: smoke-test that the entry point instances are constructable ─────
# This is a separate, clearly labelled test that CAN fail if there is an
# import problem — and that is useful information. But it is isolated from
# the pattern tests above so a circular import only breaks this one test.

class TestEntryPointConstruction:

    def test_jv_entry_point_name(self):
        from nomad_chose.parsers import parser_entry_point
        assert parser_entry_point.name == 'ChoseJVParser'

    # def test_eqe_entry_point_name(self):
    #     from nomad_chose.parsers import chose_eqe_parser
    #     assert chose_eqe_parser.name == 'ChoseEQEParser'

    def test_jv_entry_point_loads_parser(self):
        from nomad_chose.parsers import parser_entry_point
        from nomad_chose.parsers.jv_parser import ChoseJVParser
        assert isinstance(parser_entry_point.load(), ChoseJVParser)

    # def test_eqe_entry_point_loads_parser(self):
    #     from nomad_chose.parsers import chose_eqe_parser
    #     from nomad_chose.parsers.eqe_parser import ChoseEQEParser
    #     assert isinstance(chose_eqe_parser.load(), ChoseEQEParser)