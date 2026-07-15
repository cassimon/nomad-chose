from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from baseclasses.solar_energy import SolarCellEQECustom
from baseclasses.solar_energy.jvmeasurement import SolarCellJV

# 1 sun, AM 1.5G. The .txt instrument exports do not record the illumination, so
# this is the assumed default; a value supplied by the app (or the .csv header's
# `light_intensity`) overrides it.
DEFAULT_INTENSITY_MW_CM2 = 100.0


@dataclass
class ParsedStabilityData:
    parameters: dict[str, Any]
    tracking: dict[str, Any]


_HEADER_SECTION_RE = re.compile(r'^\[(?P<name>.+)\]$')


def parse_header_sections(text: str) -> dict[str, dict[str, str]]:
    """Parse the whole `## Header ##` block into {section: {key: value}}.

    The CHOSE exports carry several bracketed blocks — `[General info]`,
    `[JV Settings]`, `[Cell Settings]`, `[Tracking Settings]`,
    `[Acquisition Settings]`, `[Device Settings]`. `parse_measurement_metadata`
    only ever returned the first one, so everything else was invisible.

    Keys and section names are lower-cased. Entries with an empty value are
    dropped (the IPCE export pads its lines with trailing tabs).
    """
    sections: dict[str, dict[str, str]] = {}
    current = 'general info'

    for raw_line in text.splitlines():
        if raw_line.strip() == '## Data ##':
            break

        stripped = raw_line.strip()
        if not stripped or stripped == '## Header ##':
            continue

        match = _HEADER_SECTION_RE.match(stripped)
        if match:
            current = match.group('name').strip().lower()
            sections.setdefault(current, {})
            continue

        if '\t' not in raw_line:
            continue

        key, value = raw_line.split('\t', 1)
        key = key.strip().lower()
        value = value.strip()
        if key and value:
            sections.setdefault(current, {})[key] = value

    return sections


def _header_datetime(general: dict[str, str]) -> datetime | None:
    """Combine the header's `Date` and `Time` into the measurement timestamp.

    Without this the entry's `datetime` silently defaults to the time of upload.
    """
    date = (general.get('date') or '').strip()
    time = (general.get('time') or '').strip()
    if not date:
        return None
    try:
        return datetime.fromisoformat(f'{date}T{time}' if time else date)
    except ValueError:
        return None


def _area_cm2(general: dict[str, str], cell: dict[str, str]) -> float | None:
    for source in (general, cell):
        for key in ('cell area (cm2)', 'cell area (cm²)'):
            value = _safe_float(source.get(key, ''))
            if value:
                return value
    return None


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
    # UV-Vis transmittance export: no "## Header ##" block, a "<name> - RawData"
    # first line and a "Wavelength nm. T%" column header, then space-separated
    # "<wavelength> <T%>" rows.
    if suffix.endswith('.txt') and 'Wavelength nm. T%' in text:
        return 'uvvis'
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


def _parameter_indices(
    headers: list[str], rows: list[list[str]]
) -> dict[str, int]:
    """Column indices for the Stability (Parameters) table.

    The instrument writes a *second, unnamed* `Time (Hours)` column at the start
    of the RV block, so the data rows carry one more field than the header row
    names. Mapping RV names by their header position therefore reads one column
    too early: `Fill Factor (%) RV` would return R//, and `Efficiency (%) RV`
    would return the fill factor. Detect the extra column and shift the RV block.
    """
    indices = {name: idx for idx, name in enumerate(headers)}
    width = max((len(row) for row in rows), default=len(headers))
    if width != len(headers) + 1:
        return indices

    first_rv = next(
        (idx for idx, name in enumerate(headers) if name.strip().endswith('RV')),
        None,
    )
    if first_rv is None:
        return indices

    for name, idx in list(indices.items()):
        if idx >= first_rv:
            indices[name] = idx + 1
    return indices


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


# ── The JV archive dict ───────────────────────────────────────────────────────
#
# `build_jv_dict` produces the dict that
# `baseclasses.helper.archive_builder.jv_archive.get_jv_archive` consumes — the
# same contract nomad-hysprint feeds. It is applied to the measurement by
# `apply_jv_dict` below.

_SUMMARY_HEADER = 'Scan\tVoc\tJsc\tV_MPP\tJ_MPP\tP_MPP\tRs\tR//\tFF\tEff'
_CURVE_HEADER_PREFIX = 'V_FW (V)\tJ_FW (mA/cm'

# Column order of the `## Data ##` summary table, after the leading `Scan` label.
_SUMMARY_COLUMNS = (
    'voc',
    'jsc',
    'v_mpp',
    'j_mpp',
    'p_mpp',
    'r_ser',
    'r_par',
    'ff_percent',
    'eff_percent',
)


def _parse_jv_summary(lines: list[str]) -> dict[str, dict[str, float]]:
    """Read the per-scan summary table (FW / RV rows) of a Stability (JV) export.

    Every column is kept. The previous reader dropped V_MPP, J_MPP, Rs and R//.
    """
    header_index = None
    for idx, line in enumerate(lines):
        if line.startswith(_SUMMARY_HEADER):
            header_index = idx
            break
    if header_index is None:
        return {}

    summary: dict[str, dict[str, float]] = {}
    for line in lines[header_index + 2 :]:  # +2 skips the units row
        if not line.strip():
            continue
        if line.startswith(_CURVE_HEADER_PREFIX):
            break
        parts = [part.strip() for part in line.split('\t') if part.strip()]
        if len(parts) < len(_SUMMARY_COLUMNS) + 1:
            continue
        values = {}
        for name, raw in zip(_SUMMARY_COLUMNS, parts[1:]):
            values[name] = _safe_float(raw)
        summary[parts[0]] = values
    return summary


def _parse_jv_curves(lines: list[str]) -> dict[str, dict[str, np.ndarray]]:
    """Read the V/J columns for the forward and reverse scans."""
    header_index = None
    for idx, line in enumerate(lines):
        if line.startswith(_CURVE_HEADER_PREFIX):
            header_index = idx
            break
    if header_index is None:
        return {}

    columns: dict[str, tuple[list[float], list[float]]] = {
        'FW': ([], []),
        'RV': ([], []),
    }
    for line in lines[header_index + 1 :]:
        if not line.strip():
            continue
        cols = [part.strip() for part in line.split('\t')]
        for scan, (v_idx, j_idx) in (('FW', (0, 1)), ('RV', (2, 3))):
            if len(cols) <= j_idx:
                continue
            voltage = _safe_float(cols[v_idx])
            current = _safe_float(cols[j_idx])
            if voltage is None or current is None:
                continue
            columns[scan][0].append(voltage)
            columns[scan][1].append(current)

    curves: dict[str, dict[str, np.ndarray]] = {}
    for scan, (voltages, currents) in columns.items():
        if voltages and currents:
            curves[scan] = {
                'voltage': np.asarray(voltages, dtype=np.float64),
                'current_density': np.asarray(currents, dtype=np.float64),
            }
    return curves


def build_jv_dict(
    text: str,
    filename: str | None = None,
    intensity: float | None = None,
    logger=None,
) -> dict[str, Any] | None:
    """Build the JV archive dict for a CHOSE JV export, or None if not a JV file.

    Values that the file simply does not state are left as None rather than
    guessed; `apply_jv_dict` skips them.
    """
    kind = _detect_measurement_kind_from_text(text, filename)
    if kind not in {'jv_csv', 'stability_jv'}:
        return None

    if kind == 'jv_csv':
        return _build_jv_dict_csv(text, intensity, logger)
    return _build_jv_dict_stability(text, intensity, logger)


def _build_jv_dict_csv(
    text: str, intensity: float | None, logger=None
) -> dict[str, Any] | None:
    header: dict[str, str] = {}
    voltages: list[float] = []
    currents: list[float] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith('#'):
            key, _, value = line[1:].partition(':')
            if value:
                header[key.strip().lower()] = value.strip()
            continue
        if not line or line == 'voltage,current_density':
            continue
        values = line.split(',')
        if len(values) < 2:
            continue
        voltage = _safe_float(values[0])
        current = _safe_float(values[1])
        if voltage is None or current is None:
            if logger:
                logger.warning(f'build_jv_dict: skipping malformed row {line!r}')
            continue
        voltages.append(voltage)
        currents.append(current)

    if not voltages:
        return None

    # The csv variant is the only format that records the illumination itself.
    file_intensity = _safe_float(header.get('light_intensity', ''))
    dark = str(header.get('dark', '')).strip().lower() in {'1', 'true', 'yes'}

    settings = dict(header)
    return {
        'datetime': None,
        'active_area': _safe_float(header.get('active_area', '')) or _safe_float(header.get('cell_area', '')),
        'intensity': (
            0.0 if dark else (file_intensity or intensity or DEFAULT_INTENSITY_MW_CM2)
        ),
        'jv_curve': [
            {
                'name': str(header.get('scan') or 'JV'),
                'voltage': np.asarray(voltages, dtype=np.float64),
                'current_density': np.asarray(currents, dtype=np.float64),
                'dark': dark,
                # No summary table: baseclasses' SolarCellJVCurve.normalize
                # derives Voc/Jsc/FF/efficiency from the curve itself.
            }
        ],
        'settings': settings,
    }


def _build_jv_dict_stability(
    text: str, intensity: float | None, logger=None
) -> dict[str, Any] | None:
    lines = text.splitlines()
    sections = parse_header_sections(text)
    general = sections.get('general info', {})
    jv_settings = sections.get('jv settings', {})
    cell_settings = sections.get('cell settings', {})

    curves = _parse_jv_curves(lines)
    if not curves:
        if logger:
            logger.warning('build_jv_dict: no JV curve table found')
        return None

    summary = _parse_jv_summary(lines)
    active_area = _area_cm2(general, cell_settings)

    jv_curve: list[dict[str, Any]] = []
    for scan in ('FW', 'RV'):
        curve = curves.get(scan)
        if curve is None:
            continue
        entry: dict[str, Any] = {
            'name': scan,
            'voltage': curve['voltage'],
            'current_density': curve['current_density'],
            'dark': False,
        }
        values = summary.get(scan) or {}
        entry['open_circuit_voltage'] = values.get('voc')
        entry['short_circuit_current_density'] = values.get('jsc')
        entry['potential_at_maximum_power_point'] = values.get('v_mpp')
        entry['current_density_at_maximun_power_point'] = values.get('j_mpp')
        # The file states FF in percent; the schema quantity is a fraction.
        ff_percent = values.get('ff_percent')
        entry['fill_factor'] = ff_percent / 100.0 if ff_percent is not None else None
        entry['efficiency'] = values.get('eff_percent')
        # Rs and R// are reported in ohm; the schema quantity is area-normalised
        # (ohm*cm^2), so scale by the cell area.
        entry['series_resistance'] = _area_normalised(values.get('r_ser'), active_area)
        entry['shunt_resistance'] = _area_normalised(values.get('r_par'), active_area)
        jv_curve.append(entry)

    return {
        'datetime': _header_datetime(general),
        'active_area': active_area,
        'intensity': intensity if intensity is not None else DEFAULT_INTENSITY_MW_CM2,
        'averaging': _safe_float(jv_settings.get('averaging', '')),
        'operator': general.get('user'),
        'device': general.get('device'),
        'note': general.get('note'),
        'jv_curve': jv_curve,
        'settings': {
            **{f'jv:{k}': v for k, v in jv_settings.items()},
            **{f'cell:{k}': v for k, v in cell_settings.items()},
        },
    }


def _area_normalised(resistance: float | None, area_cm2: float | None) -> float | None:
    """ohm -> ohm*cm^2. Without the area we cannot normalise, so state nothing."""
    if resistance is None or not area_cm2:
        return None
    return resistance * area_cm2


# Units of the SolarCellJVCurve quantities, applied when writing the archive.
_CURVE_UNITS = {
    'open_circuit_voltage': 'V',
    'short_circuit_current_density': 'mA/cm^2',
    'potential_at_maximum_power_point': 'V',
    'current_density_at_maximun_power_point': 'mA/cm^2',
    'series_resistance': 'ohm*cm^2',
    'shunt_resistance': 'ohm*cm^2',
    'fill_factor': None,
    'efficiency': None,
}


def apply_jv_dict(jv_dict: dict[str, Any], mainfile: str | None, jvm) -> None:
    """Write a jv_dict onto a JVMeasurement — the `jv_curve` subsection.

    This mirrors `baseclasses.helper.archive_builder.jv_archive.get_jv_archive`
    (the mapper nomad-hysprint uses) with one difference: values the file does
    not state stay unset instead of raising. Upstream indexes and rounds every
    parameter unconditionally, so a format without a summary table (the .csv
    export) cannot go through it.

    Filling `jv_curve` is what makes `JVMeasurement.normalize` populate
    `archive.results.properties.optoelectronic.solar_cell` — the "Solar Cell
    Properties" panel — and what the sample schema reads to derive its JV
    defaults. Writing to `self.results` instead (as this plugin used to) leaves
    both empty.
    """
    from nomad.units import ureg

    from baseclasses.solar_energy.jvmeasurement import (
        SolarCellJVCurveCustom,
        SolarCellJVCurveDarkCustom,
    )

    if mainfile:
        jvm.file_name = Path(mainfile).name
    if jv_dict.get('datetime'):
        jvm.datetime = jv_dict['datetime']

    for attribute in (
        'active_area',
        'intensity',
        'integration_time',
        'settling_time',
        'averaging',
        'compliance',
    ):
        value = jv_dict.get(attribute)
        if value is not None:
            setattr(jvm, attribute, value)

    intensity = jv_dict.get('intensity')

    curves = []
    for curve in jv_dict.get('jv_curve') or []:
        if curve.get('dark'):
            curves.append(
                SolarCellJVCurveDarkCustom(
                    cell_name=curve.get('name'),
                    voltage=curve.get('voltage'),
                    current_density=curve.get('current_density'),
                    dark=True,
                )
            )
            continue

        jv_set = SolarCellJVCurveCustom(
            cell_name=curve.get('name'),
            voltage=curve.get('voltage'),
            current_density=curve.get('current_density'),
            dark=False,
        )
        if intensity is not None:
            jv_set.light_intensity = intensity
        for key, unit in _CURVE_UNITS.items():
            value = curve.get(key)
            if value is None:
                continue
            value = round(float(value), 8)
            setattr(jv_set, key, value * ureg(unit) if unit else value)
        curves.append(jv_set)

    jvm.jv_curve = curves



def parse_stability_parameters(filepath: str) -> dict[str, Any]:
    text = read_text(filepath)
    headers, rows = _parse_table_after_data_marker(text)
    if not headers or not rows:
        return {}

    column_map = _parameter_indices(headers, rows)

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

    # Not a plain positional map: the RV block carries an extra, unnamed Time
    # column. See _parameter_indices.
    column_map = _parameter_indices(headers, rows)

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

    def first_available_index(candidates: list[str]) -> int | None:
        for name in candidates:
            idx = column_map.get(name)
            if idx is not None:
                return idx
        lower_map = {name.strip().lower(): idx for name, idx in column_map.items()}
        for name in candidates:
            idx = lower_map.get(name.strip().lower())
            if idx is not None:
                return idx
        # Fallback for encoding variants (e.g. cm2/cm²/cm�/garbled chars).
        for name, idx in lower_map.items():
            if any(token in name for token in ['current density', 'power', 'time (hours)', 'voltage (v)']):
                # caller will validate semantic match by candidate keyword below
                pass
        return None

    def index_by_keyword(keyword: str) -> int | None:
        for name, idx in column_map.items():
            if keyword in name.strip().lower():
                return idx
        return None

    def column_values(names: str | list[str], keyword: str | None = None) -> np.ndarray:
        candidates = [names] if isinstance(names, str) else names
        idx = first_available_index(candidates)
        if idx is None and keyword is not None:
            idx = index_by_keyword(keyword)
        if idx is None:
            return np.asarray([], dtype=np.float64)
        values = [_safe_float(row[idx]) for row in rows if idx < len(row)]
        valid = [value for value in values if value is not None]
        return np.asarray(valid, dtype=np.float64)

    return {
        'time_hours': column_values('Time (Hours)', keyword='time (hours)'),
        'voltage': column_values('Voltage (V)', keyword='voltage'),
        'current_density': column_values(
            ['Current Density (mA/cm2)', 'Current Density (mA/cm²)', 'Current Density (mA/cm�)'],
            keyword='current density',
        ),
        'power': column_values(
            ['Power (mW/cm2)', 'Power (mW/cm²)', 'Power (mW/cm�)'],
            keyword='power',
        ),
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


_PARAMETER_COLUMNS = {
    'voc_fw': 'Voc (V) FW',
    'jsc_fw': 'Jsc (mA/cm2) FW',
    'v_mpp_fw': 'V_MPP (V) FW',
    'j_mpp_fw': 'J_MPP (mA/cm2) FW',
    'p_mpp_fw': 'P_MPP (mW/cm2) FW',
    'ff_fw': 'Fill Factor (%) FW',
    'efficiency_fw': 'Efficiency (%) FW',
    'voc_rv': 'Voc (V) RV',
    'jsc_rv': 'Jsc (mA/cm2) RV',
    'v_mpp_rv': 'V_MPP (V) RV',
    'j_mpp_rv': 'J_MPP (mA/cm2) RV',
    'p_mpp_rv': 'P_MPP (mW/cm2) RV',
    'ff_rv': 'Fill Factor (%) RV',
    'efficiency_rv': 'Efficiency (%) RV',
}

_SECONDS_PER_HOUR = 3600.0
_SECONDS_PER_MINUTE = 60.0


def build_mppt_dict(
    parameters_text: str | None,
    tracking_text: str | None,
    intensity: float | None = None,
    logger=None,
) -> dict[str, Any]:
    """Build the stability dict in the units `MPPTracking` actually declares.

    `MPPTracking.time` is in **seconds** and `power_density` in mW/cm^2; the file
    reports hours. Filling those native quantities (rather than the parallel
    `time_hours` / `tracking_*` ones this plugin used to declare) is what makes
    baseclasses compute the T80/T95 figures of merit and draw its PlotlyFigure.
    """
    result: dict[str, Any] = {}
    source_text = tracking_text or parameters_text or ''
    sections = parse_header_sections(source_text)
    general = sections.get('general info', {})
    cell_settings = sections.get('cell settings', {})
    tracking_settings = sections.get('tracking settings', {})

    result['datetime'] = _header_datetime(general)
    result['operator'] = general.get('user')
    result['device'] = general.get('device')
    result['note'] = general.get('note')
    result['active_area'] = _area_cm2(general, cell_settings)
    result['settings'] = {
        **{f'tracking:{k}': v for k, v in tracking_settings.items()},
        **{f'cell:{k}': v for k, v in cell_settings.items()},
    }

    use_intensity = (
        intensity if intensity is not None else DEFAULT_INTENSITY_MW_CM2
    )
    result['intensity'] = use_intensity

    if tracking_text:
        tracking = parse_stability_tracking_from_text(tracking_text)
        time_hours = tracking.get('time_hours')
        if time_hours is not None and len(time_hours):
            result['time'] = np.asarray(time_hours, dtype=np.float64) * _SECONDS_PER_HOUR
        power = tracking.get('power')
        if power is not None and len(power):
            power_density = np.asarray(power, dtype=np.float64)
            result['power_density'] = power_density
            if use_intensity:
                # Efficiency in %: the tracked power relative to the incident power.
                #
                # The sign is kept. The instrument's convention (cross-checked
                # against the Stability (JV) export, where the generating
                # quadrant has J_MPP > 0 and P_MPP > 0) is that positive power is
                # power *delivered* by the cell. A fixed-voltage track starts
                # above Voc, where the cell is driven and consumes power -- taking
                # the absolute value there would report that as a large positive
                # efficiency.
                result['efficiency'] = power_density / use_intensity * 100.0
        for key, target in (('voltage', 'voltage'), ('current_density', 'current_density')):
            values = tracking.get(key)
            if values is not None and len(values):
                result[target] = np.asarray(values, dtype=np.float64)

        duration = _safe_float(tracking_settings.get('test duration (min)', ''))
        if duration is not None:
            result['total_time'] = duration * _SECONDS_PER_MINUTE
        step = _safe_float(tracking_settings.get('dv track (v)', ''))
        if step is not None:
            result['perturbation_voltage'] = step
        delay = _safe_float(tracking_settings.get('track delay (s)', ''))
        if delay is not None:
            result['perturbation_delay'] = delay
        result['algorithm'] = tracking_settings.get('algorithm')

    if parameters_text:
        headers, rows = _parse_table_after_data_marker(parameters_text)
        column_map = _parameter_indices(headers, rows)

        def column(name: str) -> np.ndarray:
            idx = column_map.get(name)
            if idx is None:
                return np.asarray([], dtype=np.float64)
            values = [_safe_float(row[idx]) for row in rows if idx < len(row)]
            return np.asarray(
                [value for value in values if value is not None], dtype=np.float64
            )

        parameters: dict[str, np.ndarray] = {'time_hours': column('Time (Hours)')}
        for key, header in _PARAMETER_COLUMNS.items():
            values = column(header)
            if len(values):
                parameters[key] = values
        result['parameters'] = parameters

    return result


def build_eqe_dict(text: str, logger=None) -> dict[str, Any] | None:
    """Build the EQE dict, including the columns and settings previously dropped."""
    headers, rows = _parse_table_after_data_marker(text)
    if not headers or not rows:
        if logger:
            logger.warning('build_eqe_dict: no data table in provided content')
        return None

    column_map = {name: idx for idx, name in enumerate(headers)}

    def column(name: str) -> list[float | None]:
        idx = column_map.get(name)
        if idx is None:
            return []
        return [_safe_float(row[idx]) if idx < len(row) else None for row in rows]

    wavelengths = column('Wavelength (nm)')
    ipce_values = column('IPCE (%)')
    if not wavelengths or not ipce_values:
        if logger:
            logger.warning('build_eqe_dict: missing wavelength/IPCE columns')
        return None

    keep = [
        index
        for index, (wl, ipce) in enumerate(zip(wavelengths, ipce_values))
        if wl is not None and ipce is not None
    ]
    if not keep:
        return None

    wavelength_array = np.asarray([wavelengths[i] for i in keep], dtype=np.float64)
    eqe_array = np.asarray([ipce_values[i] for i in keep], dtype=np.float64) / 100.0
    photon_energy = 1239.841984 / wavelength_array

    # Sort by *ascending photon energy* -- the instrument sweeps the wavelength up,
    # which is energy going down.
    #
    # This is not cosmetic. The EQE analysis in baseclasses integrates the spectrum
    # against AM1.5G with `np.interp` and `cumulative_trapezoid`, and both of those
    # require an increasing x. Fed the spectrum backwards, the integral comes out
    # *negative*: the integrated Jsc read -0.0011 mA/cm^2 instead of 18.26 -- which
    # is the value the instrument itself reports in its own `J integrated` column
    # (18.23). Order is the only difference.
    order = np.argsort(photon_energy)
    wavelength_array = wavelength_array[order]
    eqe_array = eqe_array[order]
    photon_energy = photon_energy[order]

    sections = parse_header_sections(text)
    general = sections.get('general info', {})
    acquisition = sections.get('acquisition settings', {})
    device = sections.get('device settings', {})

    result: dict[str, Any] = {
        'datetime': _header_datetime(general),
        'operator': general.get('user'),
        'device': general.get('device'),
        'note': general.get('note'),
        'active_area': _area_cm2(general, {}),
        'temperature': _safe_float(general.get('temperature', '')),
        'wavelength_array': wavelength_array,
        'photon_energy_array': photon_energy,
        'eqe_array': eqe_array,
        'light_bias': _safe_float(acquisition.get('bias voltage (v)', '')),
        'integration_time': _safe_float(acquisition.get('acquisition time (s)', '')),
        'averaging': _safe_float(acquisition.get('averaging', '')),
        'delay': _safe_float(acquisition.get('delay (s)', '')),
        'led_level': _safe_float(acquisition.get('led level', '')),
        'chopper_frequency': _safe_float(acquisition.get('chopper frequency (hz)', '')),
        'settings': {
            **{f'acquisition:{k}': v for k, v in acquisition.items()},
            **{f'device:{k}': v for k, v in device.items()},
        },
    }

    # Columns the previous reader ignored entirely. They are per-wavelength, so
    # they take the same rows and the same order as the spectrum above -- reordering
    # only the spectrum would silently pair each current with the wrong wavelength.
    for key, header in (
        ('current_density_device', 'J device (mA/cm2)'),
        ('current_density_integrated', 'J integrated (mA/cm2)'),
    ):
        values = column(header)
        if values and any(value is not None for value in values):
            kept = [values[i] if i < len(values) else None for i in keep]
            result[key] = np.asarray(
                [np.nan if value is None else value for value in kept],
                dtype=np.float64,
            )[order]

    return result


def build_uvvis_dict(text: str, logger=None) -> dict[str, Any] | None:
    """Parse a CHOSE UV-Vis transmittance export.

    The file is a two-line header followed by space-separated "<wavelength> <T%>"
    rows::

        T-PVK 1.68 V_1.6 M_AS 100 uL - RawData
        Wavelength nm. T%
        300.0 0.1
        ...

    Returns the film ``name`` and the ``wavelength`` (nm) / ``transmittance`` (%)
    arrays, or None when no data rows parse.
    """
    lines = text.splitlines()
    name = None
    wavelengths: list[float] = []
    transmittances: list[float] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        # The film name is the first line, minus the " - RawData" suffix.
        if name is None and not line[0].isdigit():
            if line.lower().startswith('wavelength'):
                continue
            name = re.sub(r'\s*-\s*RawData\s*$', '', line).strip() or None
            continue
        # Column-header line ("Wavelength nm. T%") — skip.
        if line.lower().startswith('wavelength'):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        wl = _safe_float(parts[0])
        t = _safe_float(parts[1])
        if wl is None or t is None:
            continue
        wavelengths.append(wl)
        transmittances.append(t)

    if not wavelengths:
        if logger:
            logger.warning('build_uvvis_dict: no data rows in provided content')
        return None

    return {
        'name': name,
        'wavelength': np.asarray(wavelengths, dtype=np.float64),
        'transmittance': np.asarray(transmittances, dtype=np.float64),
    }


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
