"""Everything the CHOSE instrument files state must reach the NOMAD schema.

The header blocks were parsed and then dropped, the JV curves were written to the
wrong subsection, and the stability track was written to quantities baseclasses
never reads. These tests pin each of those to the real files in tests/data.
"""

import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from nomad.datamodel import EntryArchive, EntryMetadata

from nomad_chose.parsers.file_reading import (
    build_eqe_dict,
    build_jv_dict,
    build_mppt_dict,
    parse_header_sections,
    read_text,
)
from nomad_chose.schema_packages.schema_package import (
    LabEQEMeasurement,
    LabJVMeasurement,
    LabStabilityMeasurement,
)

DATA = Path(__file__).parent.parent / 'data'

JV_FILE = '0001_2025-11-20_17.32.31_Stability (JV)_AI03-1A.txt'
PARAMETERS_FILE = '0000_2025-11-20_17.32.31_Stability (Parameters)_AI03-1A.txt'
TRACKING_FILE = '0000_2025-11-20_17.32.31_Stability (Tracking)_AI03-1A.txt'
IPCE_FILE = '2025-11-20_15.49.56_IPCE_AI03.txt'


def archive_for(filename):
    archive = EntryArchive(
        metadata=EntryMetadata(entry_name=filename, upload_id='u', entry_id='e')
    )
    archive.m_context = SimpleNamespace(raw_path=os.path.abspath(str(DATA)))
    return archive


def normalize(measurement, filename):
    archive = archive_for(filename)
    archive.data = measurement
    measurement.normalize(archive, logging.getLogger('test'))
    return archive


# ── Header ────────────────────────────────────────────────────────────────────


def test_every_header_block_is_read_not_just_general_info():
    sections = parse_header_sections(read_text(DATA / JV_FILE))

    assert sections['general info']['user'] == 'FDN'
    assert sections['general info']['cell area (cm2)'] == '0.09'
    # These blocks were previously invisible: parsing stopped at [JV Settings].
    assert sections['jv settings']['scan rate (mv/s)'] == '200.000'
    assert sections['jv settings']['scan direction'] == 'FW then RV'
    assert sections['cell settings']['tipology'] == 'Cell'


def test_ipce_header_survives_its_trailing_tabs():
    sections = parse_header_sections(read_text(DATA / IPCE_FILE))
    assert sections['general info']['user'] == 'FDN DE NICOLA'
    assert sections['acquisition settings']['chopper frequency (hz)'] == '17.07'
    assert sections['device settings']['current range'] == '1 mA'


# ── JV ────────────────────────────────────────────────────────────────────────


def test_jv_curves_land_in_jv_curve_and_carry_every_summary_column():
    measurement = LabJVMeasurement()
    measurement.jv_file = JV_FILE
    normalize(measurement, JV_FILE)

    assert [curve.cell_name for curve in measurement.jv_curve] == ['FW', 'RV']
    forward, reverse = measurement.jv_curve

    assert forward.open_circuit_voltage.magnitude == pytest.approx(0.530612)
    assert forward.short_circuit_current_density.magnitude == pytest.approx(15.754854)
    assert forward.fill_factor == pytest.approx(0.2538)  # file states 25.38 %
    assert forward.efficiency == pytest.approx(2.12)
    # V_MPP, J_MPP, Rs and R// were parsed and then discarded before.
    assert forward.potential_at_maximum_power_point.magnitude == pytest.approx(0.253178)
    assert forward.current_density_at_maximun_power_point.magnitude == pytest.approx(
        8.381736
    )
    assert reverse.efficiency == pytest.approx(3.67)


def test_resistances_are_area_normalised():
    """The file reports ohm; the schema quantity is ohm*cm^2."""
    measurement = LabJVMeasurement()
    measurement.jv_file = JV_FILE
    normalize(measurement, JV_FILE)

    forward = measurement.jv_curve[0]
    # 4.237019E+2 ohm * 0.09 cm^2
    assert forward.series_resistance.magnitude == pytest.approx(423.7019 * 0.09, rel=1e-3)
    assert forward.shunt_resistance.magnitude == pytest.approx(533.1396 * 0.09, rel=1e-3)


def test_header_fills_area_intensity_timestamp_and_operator():
    measurement = LabJVMeasurement()
    measurement.jv_file = JV_FILE
    normalize(measurement, JV_FILE)

    assert measurement.active_area.magnitude == pytest.approx(0.09)
    # Not in the file: 1 sun is the documented default.
    assert measurement.intensity.magnitude == pytest.approx(100.0)
    assert measurement.operator == 'FDN'
    assert measurement.lab_id == 'AI03-1A'
    # The measurement's own timestamp, not the time of upload.
    assert measurement.datetime.year == 2025
    assert (measurement.datetime.month, measurement.datetime.day) == (11, 20)
    assert measurement.datetime.hour == 17


def test_scan_settings_are_kept():
    measurement = LabJVMeasurement()
    measurement.jv_file = JV_FILE
    normalize(measurement, JV_FILE)

    settings = measurement.settings
    assert settings.scan_rate.magnitude == pytest.approx(200.0)
    assert settings.voltage_step.magnitude == pytest.approx(20.0)
    assert settings.scan_direction == 'FW then RV'
    assert settings.typology == 'Cell'


def test_solar_cell_properties_get_the_best_scan():
    """The panel that stayed empty: it is filled by JVMeasurement.normalize from
    jv_curve, so it only works once jv_curve is populated."""
    measurement = LabJVMeasurement()
    measurement.jv_file = JV_FILE
    archive = normalize(measurement, JV_FILE)

    solar_cell = archive.results.properties.optoelectronic.solar_cell
    assert solar_cell.efficiency == pytest.approx(3.67)  # the RV scan
    assert solar_cell.open_circuit_voltage.to('V').magnitude == pytest.approx(0.539999)
    assert solar_cell.fill_factor == pytest.approx(0.3261)
    assert solar_cell.illumination_intensity.to('mW/cm**2').magnitude == pytest.approx(
        100.0
    )


def test_a_supplied_intensity_overrides_the_default():
    measurement = LabJVMeasurement()
    measurement.jv_file = JV_FILE
    measurement.intensity = 80.0
    normalize(measurement, JV_FILE)

    assert measurement.intensity.magnitude == pytest.approx(80.0)
    assert measurement.jv_curve[0].light_intensity.to(
        'mW/cm**2'
    ).magnitude == pytest.approx(80.0)


# ── Stability ─────────────────────────────────────────────────────────────────


def test_stability_fills_the_native_mpptracking_quantities():
    measurement = LabStabilityMeasurement()
    measurement.stability_parameters_file = PARAMETERS_FILE
    measurement.stability_tracking_file = TRACKING_FILE
    normalize(measurement, TRACKING_FILE)

    # `time` is seconds in the schema; the file states hours.
    assert measurement.time.to('s').magnitude[0] == pytest.approx(
        0.002487 * 3600, rel=1e-4
    )
    assert measurement.power_density.magnitude[0] == pytest.approx(-17.572114)
    assert measurement.voltage.magnitude[0] == pytest.approx(1.399999)
    assert measurement.active_area.magnitude == pytest.approx(0.09)


def test_tracking_settings_land_in_mpptracking_properties():
    measurement = LabStabilityMeasurement()
    measurement.stability_tracking_file = TRACKING_FILE
    normalize(measurement, TRACKING_FILE)

    assert measurement.properties.perturbation_voltage.magnitude == pytest.approx(0.02)
    assert measurement.properties.time.to('s').magnitude == pytest.approx(60.0)
    assert measurement.algorithm == 'Fixed Voltage'


def test_the_reverse_scan_parameters_are_not_shifted_by_the_unnamed_time_column():
    """The Parameters table has one more data column than header names: the RV block
    is preceded by a second, unnamed `Time (Hours)`. Reading RV by header position
    therefore returned R// as the fill factor and the FF as the efficiency.

    Cross-checked against the *other* file: the Stability (JV) export's RV summary
    row states Voc=0.539999, Jsc=20.841486, FF=32.61 %, Eff=3.67 %.
    """
    data = build_mppt_dict(read_text(DATA / PARAMETERS_FILE), None)
    parameters = data['parameters']

    assert parameters['voc_rv'][0] == pytest.approx(0.5399993)
    assert parameters['jsc_rv'][0] == pytest.approx(20.84149)
    assert parameters['ff_rv'][0] == pytest.approx(32.6144)
    assert parameters['efficiency_rv'][0] == pytest.approx(3.67055)

    # The forward block is not shifted and must stay correct.
    assert parameters['voc_fw'][0] == pytest.approx(0.5306116)
    assert parameters['efficiency_fw'][0] == pytest.approx(2.122071)


def test_a_short_track_does_not_crash_the_figures_of_merit():
    """baseclasses filters the track with a Savitzky-Golay window of len//5 at
    polyorder 3, so a track under 20 points makes scipy raise."""
    measurement = LabStabilityMeasurement()
    measurement.stability_tracking_file = TRACKING_FILE  # a single data point
    archive = normalize(measurement, TRACKING_FILE)

    assert archive is not None
    assert measurement.power_density is not None  # kept, despite no figures of merit


# ── EQE ───────────────────────────────────────────────────────────────────────


def test_eqe_runs_the_upstream_analysis():
    """SolarCellEQECustom.normalize computes the bandgap; it was never called."""
    measurement = LabEQEMeasurement()
    measurement.eqe_file = IPCE_FILE
    normalize(measurement, IPCE_FILE)

    data = measurement.eqe_data[0]
    assert data.bandgap_eqe is not None
    assert data.bandgap_eqe.to('eV').magnitude == pytest.approx(1.72, abs=0.05)
    assert len(data.eqe_array) > 10


def test_eqe_keeps_the_columns_and_settings_it_used_to_drop():
    measurement = LabEQEMeasurement()
    measurement.eqe_file = IPCE_FILE
    normalize(measurement, IPCE_FILE)

    assert measurement.temperature.to('celsius').magnitude == pytest.approx(25.57)
    assert measurement.active_area.magnitude == pytest.approx(0.1)
    assert len(measurement.current_density_device) > 10
    assert len(measurement.current_density_integrated) > 10
    assert measurement.settings.chopper_frequency.magnitude == pytest.approx(17.07)
    assert measurement.settings.current_range == '1 mA'
    assert measurement.datetime.hour == 15


def test_build_eqe_dict_rejects_a_non_measurement_file():
    assert build_eqe_dict(read_text(DATA / 'not_a_jv.txt')) is None


def test_build_jv_dict_rejects_a_non_measurement_file():
    assert build_jv_dict(read_text(DATA / 'not_a_jv.txt'), 'not_a_jv.txt') is None
