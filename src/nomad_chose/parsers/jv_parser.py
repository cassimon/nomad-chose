from typing import (
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import (
        EntryArchive,
    )
    from structlog.stdlib import (
        BoundLogger,
    )

from nomad.config import config
from nomad.datamodel.metainfo.workflow import Workflow
from nomad.parsing.parser import MatchingParser
from baseclasses.solar_energy.jvmeasurement import SolarCellJV
import re
import numpy as np
from nomad.parsing import MatchingParser
from nomad.datamodel import EntryArchive
from pathlib import Path

configuration = config.get_plugin_entry_point(
    'nomad_chose.parsers:parser_entry_point'
)


# ── Pure parsing function ─────────────────────────────────────────────────────
# Kept separate so it can be unit-tested without any NOMAD infrastructure.

def parse_jv_csv(filepath: str, logger=None) -> SolarCellJV:
    """
    Parse a CHOSE JV CSV file.

    Expected format:
        # operator: Alice
        # light_intensity: 100.0
        voltage,current_density
        0.00,21.30
        ...
        1.12,0.00

    Returns a populated SolarCellJV section, or None if no data rows found.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    header: dict[str, str] = {}
    data_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#'):
            m = re.match(r'#\s*(\w+):\s*(.+)', stripped)
            if m:
                header[m.group(1)] = m.group(2).strip()
        elif stripped and stripped != 'voltage,current_density':
            data_lines.append(stripped)

    if not data_lines:
        if logger:
            logger.warning(f'parse_jv_csv: no data rows in {filepath}')
        return None

    voltages, currents = [], []
    for line in data_lines:
        parts = line.split(',')
        try:
            voltages.append(float(parts[0]))
            currents.append(float(parts[1]))
        except (ValueError, IndexError) as e:
            if logger:
                logger.warning(f'parse_jv_csv: skipping malformed row {line!r}: {e}')

    if not voltages:
        return None

    v = np.array(voltages)
    j = np.array(currents)
    power = v * j
    light  = float(header.get('light_intensity', 100.0))

    jv = SolarCellJV()
    jv.voltage         = v
    jv.current_density = j
    jv.light_intensity = light

    # Scalar results derived from arrays
    jv.short_circuit_current_density = float(j[np.argmin(np.abs(v))])
    jv.open_circuit_voltage          = float(v[np.argmin(np.abs(j))])
    p_max = float(np.max(power))
    denom = jv.open_circuit_voltage * jv.short_circuit_current_density
    jv.fill_factor = p_max / denom if denom else None
    jv.efficiency  = (p_max / light * 100.0) if light else None

    return jv


# ── Parser class ──────────────────────────────────────────────────────────────

class ChoseJVParser(MatchingParser):
    """
    Matches CHOSE JV CSV files (*.jv.csv or *_JV_*.csv) and populates
    a LabJVMeasurement archive entry.

    The pvk_sample reference must be set either:
      - via a .archive.yaml sidecar (preferred), or
      - manually in the ELN after upload.
    """

    def parse(self, mainfile, archive, logger=None, child_archives=None):
        from nomad_chose.schema_packages.schema_package import LabJVMeasurement

        if logger is None:
            logger = logging.getLogger(__name__)

        filename = Path(mainfile).name   # ← works on Windows and Linux

        logger.info(f'ChoseJVParser: parsing {mainfile}')

        measurement = LabJVMeasurement()
        measurement.name    = filename
        measurement.jv_file = filename

        result = parse_jv_csv(mainfile, logger)
        if result is not None:
            result.data_file    = filename   # ← same fix here
            measurement.results = [result]

        archive.data = measurement