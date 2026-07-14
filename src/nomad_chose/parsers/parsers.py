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


def _stability_sibling(path: Path) -> Path | None:
    """The other half of a stability run: (Parameters) <-> (Tracking)."""
    for half, other in (
        ('(Parameters)', '(Tracking)'),
        ('(Tracking)', '(Parameters)'),
    ):
        if half in path.name:
            return path.with_name(path.name.replace(half, other))
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
    def is_mainfile(
        self,
        filename: str,
        mime: str,
        buffer: bytes,
        decoded_buffer: str,
        compression: str | None = None,
    ):
        """Skip a raw file that the Plains app has already described in an archive.

        An upload from the app carries, next to every measurement file, a
        `<filename>.archive.yaml` describing it -- and that archive is the richer
        entry: it links the sample and states the cell area and the illumination
        intensity, none of which the instrument writes into the file. Parsing the
        raw file *as well* would give the same measurement a second, poorer entry,
        and both would push a solar cell into `archive.results`, double-counting
        the device in the perovskite-database overview plots.

        Files dropped into NOMAD by hand have no such companion and still parse.
        """
        path = Path(filename)
        if Path(f'{path}.archive.yaml').is_file():
            return False

        # A stability run is two files -- (Parameters) and (Tracking) -- and they
        # are two halves of one measurement, so the app describes the pair in a
        # *single* archive, named after only one of them. The other half is spoken
        # for just the same; parsing it would recreate the very duplicate the
        # companion archive exists to prevent.
        sibling = _stability_sibling(path)
        if sibling is not None and Path(f'{sibling}.archive.yaml').is_file():
            return False

        return super().is_mainfile(filename, mime, buffer, decoded_buffer, compression)

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
        device = metadata.get('device')

        def seed(measurement):
            """Seed the header fields common to every CHOSE export.

            The rest (curves, areas, settings) is filled by the entry's own
            normalize(), which re-reads the raw file through NOMAD's context.
            """
            measurement.name = basename
            if operator:
                measurement.operator = operator
            if device:
                measurement.lab_id = device
            return measurement

        if kind in {'jv_csv', 'stability_jv'}:
            measurement = seed(LabJVMeasurement())
            measurement.jv_file = basename
            archive.data = measurement
            return

        if kind in {'stability_parameters', 'stability_tracking'}:
            measurement = seed(LabStabilityMeasurement())
            if kind == 'stability_parameters':
                measurement.stability_parameters_file = basename
                measurement.stability_tracking_file = _paired_stability_filename(mainfile, kind)
            else:
                measurement.stability_tracking_file = basename
                measurement.stability_parameters_file = _paired_stability_filename(mainfile, kind)
            archive.data = measurement
            return

        if kind == 'ipce':
            measurement = seed(LabEQEMeasurement())
            measurement.eqe_file = basename
            archive.data = measurement
            return

        logger.warning(f'ChoseParser: unsupported file format for {mainfile}')
