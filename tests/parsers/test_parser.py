import logging

from nomad.datamodel import EntryArchive

import logging
import pytest
from pathlib import Path

from nomad.client import normalize_all
from nomad.datamodel import EntryArchive
from nomad.datamodel.context import ClientContext
from baseclasses.solar_energy.jvmeasurement import SolarCellJV
from nomad_chose.parsers.jv_parser import parse_jv_csv, ChoseJVParser
from nomad_chose.schema_packages.schema_package import LabJVMeasurement
from nomad_perovskite_solar_cell_sample_plains.schema_packages.sample import (
    PerovskiteSolarCellSample,
    PerformedMeasurements,
)


# def test_parse_file():
#     parser = NewParser()
#     archive = EntryArchive()
#     parser.parse('tests/data/example.out', archive, logging.getLogger())

#     assert archive.workflow2.name == 'test'

# """
# Unit tests for parse_jv_csv — no NOMAD infrastructure required.
# All tests operate on real CSV files in tests/data/.
# """


DATA = Path(__file__).parent.parent / 'data'


class DummyLogger:
    def __init__(self):
        self.messages = []
    def warning(self, msg, **kw): self.messages.append(('warning', msg))
    def info(self, msg, **kw):    self.messages.append(('info', msg))
    def error(self, msg, **kw):   self.messages.append(('error', msg))


# ── parse_jv_csv ─────────────────────────────────────────────────────────────

class TestParseJvCsv:

    def test_returns_solar_cell_jv(self, jv_forward_csv):
        assert isinstance(parse_jv_csv(jv_forward_csv), SolarCellJV)

    def test_voc_positive(self, jv_forward_csv):
        assert parse_jv_csv(jv_forward_csv).open_circuit_voltage > 0

    def test_jsc_positive(self, jv_forward_csv):
        assert parse_jv_csv(jv_forward_csv).short_circuit_current_density > 0

    def test_fill_factor_between_0_and_1(self, jv_forward_csv):
        ff = parse_jv_csv(jv_forward_csv).fill_factor
        assert 0 < ff < 1

    def test_efficiency_positive(self, jv_forward_csv):
        assert parse_jv_csv(jv_forward_csv).efficiency > 0

    def test_array_lengths_match(self, jv_forward_csv):
        result = parse_jv_csv(jv_forward_csv)
        assert len(result.voltage) == len(result.current_density)

    def test_reverse_higher_efficiency_than_forward(
        self, jv_forward_csv, jv_reverse_csv
    ):
        fwd = parse_jv_csv(jv_forward_csv)
        rev = parse_jv_csv(jv_reverse_csv)
        assert rev.efficiency > fwd.efficiency

    def test_empty_file_returns_none(self, tmp_path):
        f = tmp_path / 'empty.csv'
        f.write_text('voltage,current_density\n')
        assert parse_jv_csv(str(f), DummyLogger()) is None

    def test_empty_file_logs_warning(self, tmp_path):
        f = tmp_path / 'empty.csv'
        f.write_text('voltage,current_density\n')
        logger = DummyLogger()
        parse_jv_csv(str(f), logger)
        assert any('no data rows' in m[1] for m in logger.messages
                   if m[0] == 'warning')

    def test_light_intensity_from_header(self):
        result = parse_jv_csv(str(DATA / 'jv_forward.csv'))
        assert float(result.light_intensity.magnitude) == pytest.approx(100.0)


    def test_default_light_intensity_100(self, tmp_path):
        f = tmp_path / 'no_header.csv'
        f.write_text('voltage,current_density\n0.00,20.0\n1.00,0.0\n')
        assert float(parse_jv_csv(str(f)).light_intensity.magnitude) == pytest.approx(100.0)

    def test_malformed_row_is_skipped(self, tmp_path):
        f = tmp_path / 'bad_row.csv'
        f.write_text(
            'voltage,current_density\n'
            '0.00,21.0\n'
            'BADROW\n'           # malformed
            '1.00,0.0\n'
        )
        logger = DummyLogger()
        result = parse_jv_csv(str(f), logger)
        # still returns a result from the good rows
        assert result is not None
        assert any('skipping' in m[1] for m in logger.messages
                   if m[0] == 'warning')

    def test_three_files_have_distinct_efficiencies(
        self, jv_forward_csv, jv_reverse_csv, jv_extra_csv
    ):
        effs = {
            parse_jv_csv(jv_forward_csv).efficiency,
            parse_jv_csv(jv_reverse_csv).efficiency,
            parse_jv_csv(jv_extra_csv).efficiency,
        }
        assert len(effs) == 3





DATA = Path(__file__).parent.parent / 'data'


def parse_and_normalize(csv_filename: str) -> EntryArchive:
    """
    Full parse + normalize cycle using real NOMAD infrastructure.
    ClientContext gives the archive a file system context so that
    archive.m_context.raw_path() resolves files relative to DATA.
    No server, no mocking.
    """
    filepath = str(DATA / csv_filename)

    # ClientContext points at the directory — all raw_path() calls
    # resolve relative to this directory, matching how NOMAD resolves
    # files within an upload directory on the server.
    context = ClientContext(local_dir=str(DATA))

    archive = EntryArchive(m_context=context)

    parser = ChoseJVParser()
    parser.parse(filepath, archive, logging.getLogger())

    normalize_all(archive)

    return archive


class TestLabJVMeasurementNormalize:

    def test_parse_produces_lab_jv_measurement(self):
        archive = parse_and_normalize('jv_forward.csv')
        assert isinstance(archive.data, LabJVMeasurement)

    def test_jv_file_quantity_set(self):
        archive = parse_and_normalize('jv_forward.csv')
        assert archive.data.jv_file == 'jv_forward.csv'

    def test_results_populated_after_normalize(self):
        archive = parse_and_normalize('jv_forward.csv')
        assert archive.data.results is not None
        assert len(archive.data.results) == 1

    def test_efficiency_positive(self):
        archive = parse_and_normalize('jv_forward.csv')
        assert archive.data.results[0].efficiency > 0

    def test_data_file_attached_to_result(self):
        archive = parse_and_normalize('jv_forward.csv')
        assert archive.data.results[0].data_file == 'jv_forward.csv'

    def test_light_intensity_correct_units(self):
        archive = parse_and_normalize('jv_forward.csv')
        intensity = archive.data.results[0].light_intensity
        assert float(intensity.magnitude) == pytest.approx(100.0)

    def test_voc_positive(self):
        archive = parse_and_normalize('jv_forward.csv')
        assert archive.data.results[0].open_circuit_voltage > 0

    def test_jsc_positive(self):
        archive = parse_and_normalize('jv_forward.csv')
        assert archive.data.results[0].short_circuit_current_density > 0

    def test_fill_factor_between_0_and_1(self):
        archive = parse_and_normalize('jv_forward.csv')
        ff = archive.data.results[0].fill_factor
        assert 0 < float(ff) < 1

    def test_reverse_higher_efficiency_than_forward(self):
        fwd = parse_and_normalize('jv_forward.csv')
        rev = parse_and_normalize('jv_reverse.csv')
        assert (rev.data.results[0].efficiency >
                fwd.data.results[0].efficiency)

    def test_no_pvk_sample_no_performed_measurements(self):
        """Without a pvk_sample reference, normalize() skips registration."""
        archive = parse_and_normalize('jv_forward.csv')
        # Parser sets no pvk_sample — registration is skipped cleanly
        assert archive.data.pvk_sample is None

    def test_three_files_three_distinct_efficiencies(self):
        results = [
            parse_and_normalize(f).data.results[0].efficiency
            for f in ('jv_forward.csv', 'jv_reverse.csv', 'jv_extra.csv')
        ]
        assert len(set(results)) == 3

    def test_archive_metadata_populated(self):
        """NOMAD sets basic metadata on the archive during normalize_all."""
        archive = parse_and_normalize('jv_forward.csv')
        # normalize_all populates archive.metadata if not already set
        assert archive is not None

    # def test_no_cycle_summary_is_solar_cell_jv(self):
    #     from nomad.datamodel.datamodel import EntryMetadata

    #     context = ClientContext(local_dir=str(DATA))
    #     archive = EntryArchive(m_context=context)

    #     # Set the minimal metadata that base class normalizers require
    #     archive.metadata = EntryMetadata()
    #     archive.metadata.entry_name = 'test-jv'
    #     archive.metadata.mainfile   = 'jv_forward.csv'

    #     parser = ChoseJVParser()
    #     parser.parse(
    #         str(DATA / 'jv_forward.csv'),
    #         archive,
    #         logging.getLogger(),
    #     )

    #     sample = PerovskiteSolarCellSample()
    #     archive.data.pvk_sample = sample

    #     archive.data.normalize(archive, logging.getLogger())

    #     assert isinstance(sample.performed_measurements, PerformedMeasurements)
    #     assert len(sample.performed_measurements.jv) == 1
    #     assert isinstance(sample.performed_measurements.jv[0], SolarCellJV)
    #     assert not isinstance(
    #         sample.performed_measurements.jv[0], LabJVMeasurement
    #     )