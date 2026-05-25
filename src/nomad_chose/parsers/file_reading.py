from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from baseclasses.solar_energy import SolarCellEQECustom
from baseclasses.solar_energy.jvmeasurement import SolarCellJV


@dataclass
class ParsedStabilityData:
    parameters: dict[str, Any]
    tracking: dict[str, Any]


def read_text(path: str | Path) -> str:
    with open(path, 'r', encoding='utf-8', errors='replace') as handle:
        return handle.read()


def detect_measurement_kind(path: str | Path) -> str | None:
    suffix = str(path).lower()
    text = read_text(path)

    if suffix.endswith('.csv') and 'voltage,current_density' in text:
        return 'jv_csv'
    if suffix.endswith('.txt') and 'Test\tStability (JV)' in text:
        return 'stability_jv'
    if suffix.endswith('.txt') and 'Test\tStability (Parameters)' in text:
        return 'stability_parameters'
    if suffix.endswith('.txt') and 'Test\tStability (Tracking)' in text:
        return 'stability_tracking'
    if suffix.endswith('.txt') and 'Test\tIPCE' in text and 'Wavelength (nm)\tIPCE (%)' in text:
        return 'ipce'
    return None


def parse_measurement_metadata(path: str | Path) -> dict[str, str]:
    suffix = str(path).lower()
    metadata: dict[str, str] = {}

    if suffix.endswith('.csv'):
        with open(path, 'r', encoding='utf-8', errors='replace') as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped.startswith('#'):
                    break
                parts = stripped[1:].split(':', 1)
                if len(parts) != 2:
                    continue
                metadata[parts[0].strip().lower()] = parts[1].strip()
        return metadata

    if suffix.endswith('.txt'):
        text = read_text(path)
        for line in text.splitlines():
            if line.strip() in {'## Data ##', '[JV Settings]', '[Tracking Settings]', '[Acquisition Settings]'}:
                break
            if '\t' not in line:
                continue
            key, value = line.split('\t', 1)
            key = key.strip().lower()
            value = value.strip()
            if key and value:
                metadata[key] = value
        return metadata

    return metadata


def _safe_float(value: str) -> float | None:
    try:
        return float(str(value).strip())
    except Exception:
        return None


def _parse_table_after_data_marker(text: str) -> tuple[list[str], list[list[str]]]:
    lines = [line.strip('\n') for line in text.splitlines()]
    data_index = None
    for idx, line in enumerate(lines):
        if line.strip() == '## Data ##':
            data_index = idx
            break
    if data_index is None or data_index + 1 >= len(lines):
        return [], []

    header_line = lines[data_index + 1]
    headers = [col.strip() for col in header_line.split('\t')]
    rows: list[list[str]] = []
    for line in lines[data_index + 2 :]:
        if not line.strip():
            continue
        rows.append([col.strip() for col in line.split('\t')])
    return headers, rows


def parse_jv_csv(filepath: str, logger=None) -> SolarCellJV | None:
    with open(filepath, 'r', encoding='utf-8', errors='replace') as handle:
        lines = handle.readlines()

    header: dict[str, str] = {}
    data_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#'):
            parts = stripped[1:].split(':', 1)
            if len(parts) == 2:
                header[parts[0].strip()] = parts[1].strip()
        elif stripped and stripped != 'voltage,current_density':
            data_lines.append(stripped)

    if not data_lines:
        if logger:
            logger.warning(f'parse_jv_csv: no data rows in {filepath}')
        return None

    voltages: list[float] = []
    currents: list[float] = []
    for line in data_lines:
        values = line.split(',')
        if len(values) < 2:
            continue
        voltage = _safe_float(values[0])
        current = _safe_float(values[1])
        if voltage is None or current is None:
            if logger:
                logger.warning(f'parse_jv_csv: skipping malformed row {line!r}')
            continue
        voltages.append(voltage)
        currents.append(current)

    if not voltages:
        return None

    return _build_jv_result(
        voltage=np.asarray(voltages, dtype=np.float64),
        current_density=np.asarray(currents, dtype=np.float64),
        light_intensity=float(header.get('light_intensity', 100.0)),
    )


def _build_jv_result(
    voltage: np.ndarray,
    current_density: np.ndarray,
    light_intensity: float = 100.0,
) -> SolarCellJV:
    jv = SolarCellJV()
    jv.voltage = voltage
    jv.current_density = current_density
    jv.light_intensity = light_intensity

    if len(voltage) > 0 and len(current_density) > 0:
        power = voltage * current_density
        jv.short_circuit_current_density = float(current_density[np.argmin(np.abs(voltage))])
        jv.open_circuit_voltage = float(voltage[np.argmin(np.abs(current_density))])
        p_max = float(np.max(power)) if len(power) else 0.0
        denom = jv.open_circuit_voltage * jv.short_circuit_current_density
        jv.fill_factor = p_max / denom if denom else None
        jv.efficiency = (p_max / light_intensity * 100.0) if light_intensity else None
    return jv


def parse_stability_jv_txt(filepath: str, logger=None) -> list[SolarCellJV]:
    text = read_text(filepath)
    lines = text.splitlines()

    summary: dict[str, dict[str, float]] = {}
    summary_header_index = None
    for idx, line in enumerate(lines):
        if line.startswith('Scan\tVoc\tJsc\tV_MPP\tJ_MPP\tP_MPP\tRs\tR//\tFF\tEff'):
            summary_header_index = idx
            break

    if summary_header_index is not None:
        for line in lines[summary_header_index + 2 :]:
            if not line.strip():
                continue
            if line.startswith('V_FW (V)\tJ_FW (mA/cm'):
                break
            parts = [part.strip() for part in line.split('\t') if part.strip()]
            if len(parts) < 10:
                continue
            scan = parts[0]
            summary[scan] = {
                'voc': float(parts[1]),
                'jsc': float(parts[2]),
                'v_mpp': float(parts[3]),
                'j_mpp': float(parts[4]),
                'p_mpp': float(parts[5]),
                'ff_percent': float(parts[8]),
                'eff_percent': float(parts[9]),
            }

    curve_header_index = None
    for idx, line in enumerate(lines):
        if line.startswith('V_FW (V)\tJ_FW (mA/cm'):
            curve_header_index = idx
            break

    if curve_header_index is None:
        if logger:
            logger.warning(f'parse_stability_jv_txt: no JV curve table in {filepath}')
        return []

    fw_v: list[float] = []
    fw_j: list[float] = []
    rv_v: list[float] = []
    rv_j: list[float] = []

    for line in lines[curve_header_index + 1 :]:
        if not line.strip():
            continue
        cols = [part.strip() for part in line.split('\t')]
        if len(cols) >= 2:
            v_fw = _safe_float(cols[0])
            j_fw = _safe_float(cols[1])
            if v_fw is not None and j_fw is not None:
                fw_v.append(v_fw)
                fw_j.append(j_fw)
        if len(cols) >= 4:
            v_rv = _safe_float(cols[2])
            j_rv = _safe_float(cols[3])
            if v_rv is not None and j_rv is not None:
                rv_v.append(v_rv)
                rv_j.append(j_rv)

    results: list[SolarCellJV] = []
    if fw_v and fw_j:
        fw = _build_jv_result(
            voltage=np.asarray(fw_v, dtype=np.float64),
            current_density=np.asarray(fw_j, dtype=np.float64),
            light_intensity=100.0,
        )
        if 'FW' in summary:
            fw.open_circuit_voltage = summary['FW']['voc']
            fw.short_circuit_current_density = summary['FW']['jsc']
            fw.fill_factor = summary['FW']['ff_percent'] / 100.0
            fw.efficiency = summary['FW']['eff_percent']
        results.append(fw)

    if rv_v and rv_j:
        rv = _build_jv_result(
            voltage=np.asarray(rv_v, dtype=np.float64),
            current_density=np.asarray(rv_j, dtype=np.float64),
            light_intensity=100.0,
        )
        if 'RV' in summary:
            rv.open_circuit_voltage = summary['RV']['voc']
            rv.short_circuit_current_density = summary['RV']['jsc']
            rv.fill_factor = summary['RV']['ff_percent'] / 100.0
            rv.efficiency = summary['RV']['eff_percent']
        results.append(rv)

    return results


def parse_jv_file(filepath: str, logger=None) -> list[SolarCellJV]:
    kind = detect_measurement_kind(filepath)
    if kind == 'jv_csv':
        result = parse_jv_csv(filepath, logger)
        return [result] if result is not None else []
    if kind == 'stability_jv':
        return parse_stability_jv_txt(filepath, logger)
    return []


def _detect_measurement_kind_from_text(text: str, filename: str | None = None) -> str | None:
    fname = str(filename).lower() if filename is not None else ''
    if fname.endswith('.csv') and 'voltage,current_density' in text:
        return 'jv_csv'
    if fname.endswith('.txt') and 'Test\tStability (JV)' in text:
        return 'stability_jv'
    if fname.endswith('.txt') and 'Test\tStability (Parameters)' in text:
        return 'stability_parameters'
    if fname.endswith('.txt') and 'Test\tStability (Tracking)' in text:
        return 'stability_tracking'
    if fname.endswith('.txt') and 'Test\tIPCE' in text and 'Wavelength (nm)\tIPCE (%)' in text:
        return 'ipce'
    return None


def parse_jv_csv_from_text(text: str, logger=None) -> SolarCellJV | None:
    lines = [line for line in text.splitlines()]

    header: dict[str, str] = {}
    data_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#'):
            parts = stripped[1:].split(':', 1)
            if len(parts) == 2:
                header[parts[0].strip()] = parts[1].strip()
        elif stripped and stripped != 'voltage,current_density':
            data_lines.append(stripped)

    if not data_lines:
        if logger:
            logger.warning('parse_jv_csv_from_text: no data rows in provided content')
        return None

    voltages: list[float] = []
    currents: list[float] = []
    for line in data_lines:
        values = line.split(',')
        if len(values) < 2:
            continue
        voltage = _safe_float(values[0])
        current = _safe_float(values[1])
        if voltage is None or current is None:
            if logger:
                logger.warning(f'parse_jv_csv_from_text: skipping malformed row {line!r}')
            continue
        voltages.append(voltage)
        currents.append(current)

    if not voltages:
        return None

    return _build_jv_result(
        voltage=np.asarray(voltages, dtype=np.float64),
        current_density=np.asarray(currents, dtype=np.float64),
        light_intensity=float(header.get('light_intensity', 100.0)),
    )


def parse_stability_jv_txt_from_text(text: str, logger=None) -> list[SolarCellJV]:
    lines = text.splitlines()

    summary: dict[str, dict[str, float]] = {}
    summary_header_index = None
    for idx, line in enumerate(lines):
        if line.startswith('Scan\tVoc\tJsc\tV_MPP\tJ_MPP\tP_MPP\tRs\tR//\tFF\tEff'):
            summary_header_index = idx
            break

    if summary_header_index is not None:
        for line in lines[summary_header_index + 2 :]:
            if not line.strip():
                continue
            if line.startswith('V_FW (V)\tJ_FW (mA/cm'):
                break
            parts = [part.strip() for part in line.split('\t') if part.strip()]
            if len(parts) < 10:
                continue
            scan = parts[0]
            summary[scan] = {
                'voc': float(parts[1]),
                'jsc': float(parts[2]),
                'v_mpp': float(parts[3]),
                'j_mpp': float(parts[4]),
                'p_mpp': float(parts[5]),
                'ff_percent': float(parts[8]),
                'eff_percent': float(parts[9]),
            }

    curve_header_index = None
    for idx, line in enumerate(lines):
        if line.startswith('V_FW (V)\tJ_FW (mA/cm'):
            curve_header_index = idx
            break

    if curve_header_index is None:
        if logger:
            logger.warning('parse_stability_jv_txt_from_text: no JV curve table in provided content')
        return []

    fw_v: list[float] = []
    fw_j: list[float] = []
    rv_v: list[float] = []
    rv_j: list[float] = []

    for line in lines[curve_header_index + 1 :]:
        if not line.strip():
            continue
        cols = [part.strip() for part in line.split('\t')]
        if len(cols) >= 2:
            v_fw = _safe_float(cols[0])
            j_fw = _safe_float(cols[1])
            if v_fw is not None and j_fw is not None:
                fw_v.append(v_fw)
                fw_j.append(j_fw)
        if len(cols) >= 4:
            v_rv = _safe_float(cols[2])
            j_rv = _safe_float(cols[3])
            if v_rv is not None and j_rv is not None:
                rv_v.append(v_rv)
                rv_j.append(j_rv)

    results: list[SolarCellJV] = []
    if fw_v and fw_j:
        fw = _build_jv_result(
            voltage=np.asarray(fw_v, dtype=np.float64),
            current_density=np.asarray(fw_j, dtype=np.float64),
            light_intensity=100.0,
        )
        if 'FW' in summary:
            fw.open_circuit_voltage = summary['FW']['voc']
            fw.short_circuit_current_density = summary['FW']['jsc']
            fw.fill_factor = summary['FW']['ff_percent'] / 100.0
            fw.efficiency = summary['FW']['eff_percent']
        results.append(fw)

    if rv_v and rv_j:
        rv = _build_jv_result(
            voltage=np.asarray(rv_v, dtype=np.float64),
            current_density=np.asarray(rv_j, dtype=np.float64),
            light_intensity=100.0,
        )
        if 'RV' in summary:
            rv.open_circuit_voltage = summary['RV']['voc']
            rv.short_circuit_current_density = summary['RV']['jsc']
            rv.fill_factor = summary['RV']['ff_percent'] / 100.0
            rv.efficiency = summary['RV']['eff_percent']
        results.append(rv)

    return results


def parse_jv_file_from_text(text: str, filename: str | None = None, logger=None) -> list[SolarCellJV]:
    kind = _detect_measurement_kind_from_text(text, filename)
    if kind == 'jv_csv':
        result = parse_jv_csv_from_text(text, logger)
        return [result] if result is not None else []
    if kind == 'stability_jv':
        return parse_stability_jv_txt_from_text(text, logger)
    return []



def parse_stability_parameters(filepath: str) -> dict[str, Any]:
    text = read_text(filepath)
    headers, rows = _parse_table_after_data_marker(text)
    if not headers or not rows:
        return {}

    column_map = {name: idx for idx, name in enumerate(headers)}

    def column_values(name: str) -> np.ndarray:
        idx = column_map.get(name)
        if idx is None:
            return np.asarray([], dtype=np.float64)
        values = [_safe_float(row[idx]) for row in rows if idx < len(row)]
        valid = [value for value in values if value is not None]
        return np.asarray(valid, dtype=np.float64)

    return {
        'time_hours': column_values('Time (Hours)'),
        'voc_fw': column_values('Voc (V) FW'),
        'jsc_fw': column_values('Jsc (mA/cm2) FW'),
        'efficiency_fw': column_values('Efficiency (%) FW'),
        'voc_rv': column_values('Voc (V) RV'),
        'jsc_rv': column_values('Jsc (mA/cm2) RV'),
        'efficiency_rv': column_values('Efficiency (%) RV'),
    }


def parse_stability_parameters_from_text(text: str) -> dict[str, Any]:
    headers, rows = _parse_table_after_data_marker(text)
    if not headers or not rows:
        return {}

    column_map = {name: idx for idx, name in enumerate(headers)}

    def column_values(name: str) -> np.ndarray:
        idx = column_map.get(name)
        if idx is None:
            return np.asarray([], dtype=np.float64)
        values = [_safe_float(row[idx]) for row in rows if idx < len(row)]
        valid = [value for value in values if value is not None]
        return np.asarray(valid, dtype=np.float64)

    return {
        'time_hours': column_values('Time (Hours)'),
        'voc_fw': column_values('Voc (V) FW'),
        'jsc_fw': column_values('Jsc (mA/cm2) FW'),
        'efficiency_fw': column_values('Efficiency (%) FW'),
        'voc_rv': column_values('Voc (V) RV'),
        'jsc_rv': column_values('Jsc (mA/cm2) RV'),
        'efficiency_rv': column_values('Efficiency (%) RV'),
    }


def parse_stability_tracking(filepath: str) -> dict[str, Any]:
    text = read_text(filepath)
    headers, rows = _parse_table_after_data_marker(text)
    if not headers or not rows:
        return {}

    column_map = {name: idx for idx, name in enumerate(headers)}

    def column_values(name: str) -> np.ndarray:
        idx = column_map.get(name)
        if idx is None:
            return np.asarray([], dtype=np.float64)
        values = [_safe_float(row[idx]) for row in rows if idx < len(row)]
        valid = [value for value in values if value is not None]
        return np.asarray(valid, dtype=np.float64)

    return {
        'time_hours': column_values('Time (Hours)'),
        'voltage': column_values('Voltage (V)'),
        'current_density': column_values('Current Density (mA/cm�)'),
        'power': column_values('Power (mW/cm�)'),
    }


def parse_stability_tracking_from_text(text: str) -> dict[str, Any]:
    headers, rows = _parse_table_after_data_marker(text)
    if not headers or not rows:
        return {}

    column_map = {name: idx for idx, name in enumerate(headers)}

    def column_values(name: str) -> np.ndarray:
        idx = column_map.get(name)
        if idx is None:
            return np.asarray([], dtype=np.float64)
        values = [_safe_float(row[idx]) for row in rows if idx < len(row)]
        valid = [value for value in values if value is not None]
        return np.asarray(valid, dtype=np.float64)

    return {
        'time_hours': column_values('Time (Hours)'),
        'voltage': column_values('Voltage (V)'),
        'current_density': column_values('Current Density (mA/cm2)'),
        'power': column_values('Power (mW/cm2)'),
    }


def parse_stability_pair_from_text(
    parameters_text: str | None,
    tracking_text: str | None,
) -> ParsedStabilityData:
    parameters = parse_stability_parameters_from_text(parameters_text) if parameters_text else {}
    tracking = parse_stability_tracking_from_text(tracking_text) if tracking_text else {}
    return ParsedStabilityData(parameters=parameters, tracking=tracking)


def parse_stability_pair(
    parameters_path: str | None,
    tracking_path: str | None,
) -> ParsedStabilityData:
    parameters = parse_stability_parameters(parameters_path) if parameters_path else {}
    tracking = parse_stability_tracking(tracking_path) if tracking_path else {}
    return ParsedStabilityData(parameters=parameters, tracking=tracking)


def parse_ipce_file(filepath: str, logger=None) -> SolarCellEQECustom | None:
    text = read_text(filepath)
    headers, rows = _parse_table_after_data_marker(text)
    if not headers or not rows:
        if logger:
            logger.warning(f'parse_ipce_file: no data table in {filepath}')
        return None

    column_map = {name: idx for idx, name in enumerate(headers)}
    wl_idx = column_map.get('Wavelength (nm)')
    ipce_idx = column_map.get('IPCE (%)')
    if wl_idx is None or ipce_idx is None:
        if logger:
            logger.warning(f'parse_ipce_file: missing wavelength/IPCE columns in {filepath}')
        return None

    wavelengths: list[float] = []
    ipce_values: list[float] = []
    for row in rows:
        if wl_idx >= len(row) or ipce_idx >= len(row):
            continue
        wl = _safe_float(row[wl_idx])
        ipce = _safe_float(row[ipce_idx])
        if wl is None or ipce is None:
            continue
        wavelengths.append(wl)
        ipce_values.append(ipce)

    if not wavelengths:
        return None

    wavelength_array = np.asarray(wavelengths, dtype=np.float64)
    photon_energy = 1239.841984 / wavelength_array
    eqe = np.asarray(ipce_values, dtype=np.float64) / 100.0

    entry = SolarCellEQECustom(
        photon_energy_array=photon_energy,
        raw_photon_energy_array=photon_energy,
        eqe_array=eqe,
        raw_eqe_array=eqe,
    )
    return entry


def parse_ipce_file_from_text(text: str, filename: str | None = None, logger=None) -> SolarCellEQECustom | None:
    headers, rows = _parse_table_after_data_marker(text)
    if not headers or not rows:
        if logger:
            logger.warning('parse_ipce_file_from_text: no data table in provided content')
        return None

    column_map = {name: idx for idx, name in enumerate(headers)}
    wl_idx = column_map.get('Wavelength (nm)')
    ipce_idx = column_map.get('IPCE (%)')
    if wl_idx is None or ipce_idx is None:
        if logger:
            logger.warning('parse_ipce_file_from_text: missing wavelength/IPCE columns')
        return None

    wavelengths: list[float] = []
    ipce_values: list[float] = []
    for row in rows:
        if wl_idx >= len(row) or ipce_idx >= len(row):
            continue
        wl = _safe_float(row[wl_idx])
        ipce = _safe_float(row[ipce_idx])
        if wl is None or ipce is None:
            continue
        wavelengths.append(wl)
        ipce_values.append(ipce)

    if not wavelengths:
        return None

    wavelength_array = np.asarray(wavelengths, dtype=np.float64)
    photon_energy = 1239.841984 / wavelength_array
    eqe = np.asarray(ipce_values, dtype=np.float64) / 100.0

    entry = SolarCellEQECustom(
        photon_energy_array=photon_energy,
        raw_photon_energy_array=photon_energy,
        eqe_array=eqe,
        raw_eqe_array=eqe,
    )
    return entry
