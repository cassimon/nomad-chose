from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from nomad.config import config
from nomad.datamodel import EntryArchive
from nomad.datamodel.metainfo.workflow import Workflow
from nomad.parsing import MatchingParser

from nomad_chose.parsers.file_reading import detect_measurement_kind, parse_measurement_metadata

if TYPE_CHECKING:
    from structlog.stdlib import BoundLogger


configuration = config.get_plugin_entry_point(
    'nomad_chose.parsers:parser_entry_point'
)


def _paired_stability_filename(mainfile: str, kind: str) -> str | None:
    path = Path(mainfile)
    name = path.name
    if kind == 'stability_parameters':
        candidate = path.with_name(name.replace('(Parameters)', '(Tracking)'))
        return candidate.name if candidate.exists() else None
    if kind == 'stability_tracking':
        candidate = path.with_name(name.replace('(Tracking)', '(Parameters)'))
        return candidate.name if candidate.exists() else None
    return None


class NewParser(MatchingParser):
    def parse(
        self,
        mainfile: str,
        archive: EntryArchive,
        logger: 'BoundLogger',
        child_archives: dict[str, EntryArchive] = None,
    ) -> None:
        logger.info('NewParser.parse', parameter=configuration.parameter)
        archive.workflow2 = Workflow(name='test')


class ChoseParser(MatchingParser):
    def parse(
        self,
        mainfile: str,
        archive: EntryArchive,
        logger,
        child_archives=None,
    ) -> None:
        from nomad_chose.schema_packages.schema_package import (
            LabEQEMeasurement,
            LabJVMeasurement,
            LabStabilityMeasurement,
        )

        logger.info(f'ChoseParser: parsing {mainfile}')
        kind = detect_measurement_kind(mainfile)
        basename = Path(mainfile).name
        metadata = parse_measurement_metadata(mainfile)
        operator = metadata.get('operator') or metadata.get('user')

        if kind in {'jv_csv', 'stability_jv'}:
            measurement = LabJVMeasurement()
            measurement.name = basename
            measurement.jv_file = basename
            if operator:
                measurement.operator = operator
            archive.data = measurement
            return

        if kind in {'stability_parameters', 'stability_tracking'}:
            measurement = LabStabilityMeasurement()
            measurement.name = basename
            if operator:
                measurement.operator = operator
            if kind == 'stability_parameters':
                measurement.stability_parameters_file = basename
                measurement.stability_tracking_file = _paired_stability_filename(mainfile, kind)
            else:
                measurement.stability_tracking_file = basename
                measurement.stability_parameters_file = _paired_stability_filename(mainfile, kind)
            archive.data = measurement
            return

        if kind == 'ipce':
            measurement = LabEQEMeasurement()
            measurement.name = basename
            measurement.eqe_file = basename
            if operator:
                measurement.operator = operator
            archive.data = measurement
            return

        logger.warning(f'ChoseParser: unsupported file format for {mainfile}')
