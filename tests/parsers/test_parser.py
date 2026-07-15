import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from baseclasses.solar_energy.jvmeasurement import SolarCellJV
from nomad.datamodel import EntryArchive
from nomad_chose.parsers.file_reading import (
    detect_measurement_kind,
    parse_ipce_file,
    parse_jv_csv,
    parse_jv_file,
    parse_stability_pair,
)
from nomad_chose.parsers.parsers import ChoseParser
from nomad_chose.schema_packages.schema_package import (
    LabEQEMeasurement,
    LabJVMeasurement,
    LabStabilityMeasurement,
    LabUVvisMeasurement,
)
from nomad_perovskite_solar_cell_sample_plains.schema_packages.sample import (
    PerovskiteSolarCellSampleArea,
)

DATA = Path(__file__).parent.parent / 'data'


class DummyLogger:
    def __init__(self):
        self.messages = []

    def warning(self, msg, **kw):
        self.messages.append(('warning', msg))

    def info(self, msg, **kw):
        self.messages.append(('info', msg))


class TestFileReading:

    def test_parse_stability_jv_txt_returns_two_scans(self, stability_jv_txt):
        results = parse_jv_file(stability_jv_txt)
        assert len(results) == 2
        assert all(isinstance(result, SolarCellJV) for result in results)

    def test_detect_measurement_kind(self, stability_parameters_txt, ipce_txt):
        assert detect_measurement_kind(stability_parameters_txt) == 'stability_parameters'
        assert detect_measurement_kind(ipce_txt) == 'ipce'

    def test_detect_uvvis(self, uvvis_txt):
        assert detect_measurement_kind(uvvis_txt) == 'uvvis'

    def test_parse_stability_pair(self, stability_parameters_txt, stability_tracking_txt):
        parsed = parse_stability_pair(stability_parameters_txt, stability_tracking_txt)
        assert parsed.parameters['efficiency_fw'].size == 1
        assert parsed.tracking['power'].size == 1

    def test_parse_ipce_file(self, ipce_txt):
        parsed = parse_ipce_file(ipce_txt)
        assert parsed is not None
        assert len(parsed.photon_energy_array) > 5
        assert len(parsed.eqe_array) > 5


class TestParserDispatch:
    def test_parse_stability_jv_creates_lab_jv_measurement(self, stability_jv_txt):
        parser = ChoseParser()
        archive = EntryArchive()
        parser.parse(stability_jv_txt, archive, logging.getLogger())

        assert isinstance(archive.data, LabJVMeasurement)
        assert archive.data.jv_file.endswith('.txt')
        assert archive.data.operator == 'FDN'

    def test_parse_stability_parameters_creates_combined_stability_measurement(
        self,
        stability_parameters_txt,
    ):
        parser = ChoseParser()
        archive = EntryArchive()
        parser.parse(stability_parameters_txt, archive, logging.getLogger())

        assert isinstance(archive.data, LabStabilityMeasurement)
        assert archive.data.stability_parameters_file.endswith('(Parameters)_AI03-1A.txt')
        assert archive.data.stability_tracking_file.endswith('(Tracking)_AI03-1A.txt')

    def test_parse_ipce_creates_eqe_measurement(self, ipce_txt):
        parser = ChoseParser()
        archive = EntryArchive()
        parser.parse(ipce_txt, archive, logging.getLogger())

        assert isinstance(archive.data, LabEQEMeasurement)
        assert archive.data.eqe_file.endswith('_IPCE_AI03.txt')
        assert archive.data.operator == 'FDN DE NICOLA'

    def test_parse_uvvis_creates_uvvis_measurement(self, uvvis_txt):
        parser = ChoseParser()
        archive = EntryArchive()
        parser.parse(uvvis_txt, archive, logging.getLogger())

        assert isinstance(archive.data, LabUVvisMeasurement)
        assert archive.data.uvvis_file == 'uvvis_transmittance.txt'


class TestSchemaNormalize:
    def _archive_with_file(self, filename: str):
        archive = MagicMock()
        archive.m_context = MagicMock()
        archive.m_context.raw_path.return_value = str(DATA / filename)
        return archive

    def test_lab_jv_measurement_normalize_for_stability_jv(self):
        """The curves must land in `jv_curve`.

        `JVMeasurement.normalize` reads that subsection (and nothing else) to fill
        archive.results.properties.optoelectronic.solar_cell. This plugin used to
        write them to `results`, which NOMAD itself rejects as the wrong definition,
        leaving the Solar Cell Properties empty.
        """
        sample = PerovskiteSolarCellSampleArea()
        measurement = LabJVMeasurement()
        measurement.jv_file = '0001_2025-11-20_17.32.31_Stability (JV)_AI03-1A.txt'
        measurement.pvk_sample = sample

        with patch.object(LabJVMeasurement.__bases__[0], 'normalize', return_value=None):
            measurement.normalize(self._archive_with_file(measurement.jv_file), DummyLogger())

        assert len(measurement.jv_curve) == 2
        assert [curve.cell_name for curve in measurement.jv_curve] == ['FW', 'RV']

    def test_lab_stability_measurement_normalize(self):
        """The MPPTracking natives, not the old parallel `tracking_*` quantities."""
        measurement = LabStabilityMeasurement()
        measurement.stability_parameters_file = (
            '0000_2025-11-20_17.32.31_Stability (Parameters)_AI03-1A.txt'
        )
        measurement.stability_tracking_file = (
            '0000_2025-11-20_17.32.31_Stability (Tracking)_AI03-1A.txt'
        )

        archive = MagicMock()
        archive.m_context = MagicMock()

        def _raw_path(name):
            return str(DATA / name)

        archive.m_context.raw_path.side_effect = _raw_path

        measurement.normalize(archive, DummyLogger())

        assert measurement.power_density is not None
        assert measurement.power_density.size == 1
        assert measurement.time is not None
        assert measurement.jv_parameters.efficiency_fw is not None
        assert measurement.jv_parameters.efficiency_fw.size == 1

    def test_lab_eqe_measurement_normalize(self):
        measurement = LabEQEMeasurement()
        measurement.eqe_file = '2025-11-20_15.49.56_IPCE_AI03.txt'

        archive = MagicMock()
        archive.m_context = MagicMock()
        archive.m_context.raw_path.return_value = str(DATA / measurement.eqe_file)

        with patch.object(LabEQEMeasurement.__bases__[0], 'normalize', return_value=None):
            measurement.normalize(archive, DummyLogger())

        assert measurement.eqe_data
        assert len(measurement.eqe_data[0].eqe_array) > 10

    def test_lab_uvvis_measurement_normalize(self):
        measurement = LabUVvisMeasurement()
        measurement.uvvis_file = 'uvvis_transmittance.txt'

        archive = MagicMock()
        archive.m_context = MagicMock()
        archive.m_context.raw_path.return_value = str(DATA / measurement.uvvis_file)

        with patch.object(
            LabUVvisMeasurement.__bases__[0], 'normalize', return_value=None
        ):
            measurement.normalize(archive, DummyLogger())

        assert measurement.measurements
        spectrum = measurement.measurements[0]
        assert len(spectrum.wavelength) == 801
        assert len(spectrum.intensity) == 801
        # Wavelength in nm, transmittance in % — the film name came from line 1.
        assert spectrum.name == 'T-PVK 1.68 V_1.6 M_AS 100 uL'

    # ── Sample history (performed_measurements) registration ─────────────────

    def test_stability_jv_writes_best_scan_to_sample_history(self):
        """Stability JV normalize picks the better scan and registers it on the sample."""
        sample = PerovskiteSolarCellSampleArea()
        measurement = LabJVMeasurement()
        measurement.jv_file = '0001_2025-11-20_17.32.31_Stability (JV)_AI03-1A.txt'
        measurement.pvk_sample = sample

        with patch.object(LabJVMeasurement.__bases__[0], 'normalize', return_value=None):
            measurement.normalize(
                self._archive_with_file(measurement.jv_file), DummyLogger()
            )

        # Two JV curves (FW + RV) parsed; only the best efficiency appears in sample history
        assert len(measurement.jv_curve) == 2
        best = max(measurement.jv_curve, key=lambda curve: curve.efficiency)
        assert best.cell_name == 'RV'
        assert best.efficiency == pytest.approx(3.67)
