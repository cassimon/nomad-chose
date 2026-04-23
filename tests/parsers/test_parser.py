import logging

from nomad.datamodel import EntryArchive

from nomad_chose.parsers.jv_parser import NewParser


def test_parse_file():
    parser = NewParser()
    archive = EntryArchive()
    parser.parse('tests/data/example.out', archive, logging.getLogger())

    assert archive.workflow2.name == 'test'

"""
Unit tests for parse_jv_csv — no NOMAD infrastructure required.
All tests operate on real CSV files in tests/data/.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from baseclasses.solar_energy.jvmeasurement import SolarCellJV
from nomad_chose.parsers.jv_parser import parse_jv_csv, ChoseJVParser
from nomad_chose.schema_packages.schema_package import LabJVMeasurement
from nomad_perovskite_solar_cell_sample_plains.schema_packages.sample import (
    PerovskiteSolarCellSample,
    PerformedMeasurements,
)

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


# ── LabJVMeasurement.normalize ────────────────────────────────────────────────

class TestLabJVMeasurementNormalize:

    def _archive(self, csv_name: str):
        archive = MagicMock()
        archive.m_context = MagicMock()
        archive.m_context.raw_path.return_value = str(DATA / csv_name)
        return archive

    def _normalize(self, meas, archive):
        with patch.object(LabJVMeasurement.__bases__[0], 'normalize',
                          return_value=None):
            meas.normalize(archive, DummyLogger())

    def test_parses_file_and_sets_results(self, jv_forward_csv):
        meas = LabJVMeasurement()
        meas.jv_file = 'jv_forward.csv'
        meas.pvk_sample = PerovskiteSolarCellSample()
        self._normalize(meas, self._archive('jv_forward.csv'))
        assert meas.results and meas.results[0].efficiency > 0

    def test_data_file_attached_to_result(self):
        meas = LabJVMeasurement()
        meas.jv_file = 'jv_forward.csv'
        meas.pvk_sample = PerovskiteSolarCellSample()
        self._normalize(meas, self._archive('jv_forward.csv'))
        assert meas.results[0].data_file == 'jv_forward.csv'

    def test_summary_written_into_sample(self):
        sample = PerovskiteSolarCellSample()
        meas = LabJVMeasurement()
        meas.jv_file = 'jv_forward.csv'
        meas.pvk_sample = sample
        self._normalize(meas, self._archive('jv_forward.csv'))
        assert isinstance(sample.performed_measurements, PerformedMeasurements)
        assert len(sample.performed_measurements.jv) == 1

    def test_summary_data_file_set(self):
        sample = PerovskiteSolarCellSample()
        meas = LabJVMeasurement()
        meas.jv_file = 'jv_forward.csv'
        meas.pvk_sample = sample
        self._normalize(meas, self._archive('jv_forward.csv'))
        assert sample.performed_measurements.jv[0].data_file == 'jv_forward.csv'

    def test_three_measurements_accumulate(self):
        sample = PerovskiteSolarCellSample()
        for csv in ('jv_forward.csv', 'jv_reverse.csv', 'jv_extra.csv'):
            meas = LabJVMeasurement()
            meas.jv_file = csv
            meas.pvk_sample = sample
            self._normalize(meas, self._archive(csv))
        assert len(sample.performed_measurements.jv) == 3

    def test_no_pvk_sample_logs_warning(self):
        logger = DummyLogger()
        meas = LabJVMeasurement()
        meas.jv_file = 'jv_forward.csv'
        meas.pvk_sample = None
        with patch.object(LabJVMeasurement.__bases__[0], 'normalize',
                          return_value=None):
            meas.normalize(self._archive('jv_forward.csv'), logger)
        assert any('no pvk_sample' in m[1] for m in logger.messages
                   if m[0] == 'warning')

    def test_no_context_skips_file_parse(self):
        meas = LabJVMeasurement()
        meas.jv_file = 'jv_forward.csv'
        meas.pvk_sample = PerovskiteSolarCellSample()
        archive = MagicMock()
        archive.m_context = None
        with patch.object(LabJVMeasurement.__bases__[0], 'normalize',
                          return_value=None):
            meas.normalize(archive, DummyLogger())
        assert not meas.results

    def test_no_cycle_sample_not_referenced_back(self):
        """Sample object must not hold a reference back to the measurement entry."""
        sample = PerovskiteSolarCellSample()
        meas = LabJVMeasurement()
        meas.jv_file = 'jv_forward.csv'
        meas.pvk_sample = sample
        self._normalize(meas, self._archive('jv_forward.csv'))
        # performed_measurements.jv[] contains SolarCellJV objects,
        # not references to LabJVMeasurement — confirm type
        for jv_item in sample.performed_measurements.jv:
            assert isinstance(jv_item, SolarCellJV)
            assert not isinstance(jv_item, LabJVMeasurement)