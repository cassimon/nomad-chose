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
from nomad.datamodel.data import Schema
from nomad.datamodel.metainfo.annotations import ELNAnnotation, ELNComponentEnum
from nomad.metainfo import Quantity, SchemaPackage
from nomad.datamodel.data import EntryData
from nomad.datamodel.metainfo.annotations import ELNAnnotation
from nomad.metainfo import Quantity, Reference, Section, SubSection, SchemaPackage

from baseclasses.solar_energy.jvmeasurement import JVMeasurement, SolarCellJV
from nomad_perovskite_solar_cell_sample_plains.schema_packages.sample import (
    PerovskiteSolarCellSample,
    PerformedMeasurements,
)

configuration = config.get_plugin_entry_point(
    'nomad_chose.schema_packages:schema_package_entry_point'
)


class NewSchemaPackage(Schema):
    name = Quantity(
        type=str, a_eln=ELNAnnotation(component=ELNComponentEnum.StringEditQuantity)
    )
    message = Quantity(type=str)

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)

        logger.info('NewSchema.normalize', parameter=configuration.parameter)
        self.message = f'Hello {self.name}!'




m_package = SchemaPackage()


class LabJVMeasurement(JVMeasurement, EntryData):
    """
    JV measurement entry for the CHOSE lab instrument.

    """
    m_def = Section(
        label='CHOSE JV Measurement',
        a_eln=dict(
            properties=dict(
                order=['name', 'pvk_sample', 'jv_file', 'datetime', 'operator']
            )
        ),
    )

    pvk_sample = Quantity(
        type=Reference(PerovskiteSolarCellSample.m_def),
        description='The PerovskiteSolarCellSample this measurement belongs to.',
        a_eln=ELNAnnotation(component='ReferenceEditQuantity'),
    )

    jv_file = Quantity(
        type=str,
        description='Raw JV CSV file from the CHOSE instrument.',
        a_eln=ELNAnnotation(
            component='FileEditQuantity',
            label='Raw JV file (.csv)',
        ),
    )
    operator = Quantity(
        type=str,
        a_eln=ELNAnnotation(component='StringEditQuantity'),
    )

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        # Parse the raw file if present and context is available
        if self.jv_file and archive is not None and archive.m_context:
            from nomad_chose.parsers.jv_parser import parse_jv_csv
            try:
                raw_path = archive.m_context.raw_path(self.jv_file)
                result = parse_jv_csv(raw_path, logger)
                if result is not None:
                    result.data_file = self.jv_file
                    self.results = [result]
            except Exception as e:
                logger.warning(
                    f'LabJVMeasurement: could not parse {self.jv_file}: {e}'
                )

        # Copy scalar summary into sample.performed_measurements (no cycle:
        # measurement → sample only, never sample → measurement entry)
        if self.pvk_sample is None:
            logger.warning('LabJVMeasurement: no pvk_sample set, skipping registration.')
            return

        if not self.results:
            return

        from nomad_perovskite_solar_cell_sample_plains.schema_packages.sample import PerformedMeasurements
        from baseclasses.solar_energy.jvmeasurement import SolarCellJV

        best = max(
            self.results,
            key=lambda r: r.efficiency if r.efficiency is not None else 0,
        )
        summary = SolarCellJV()
        summary.efficiency                    = best.efficiency
        summary.open_circuit_voltage          = best.open_circuit_voltage
        summary.short_circuit_current_density = best.short_circuit_current_density
        summary.fill_factor                   = best.fill_factor
        summary.light_intensity               = best.light_intensity
        summary.data_file                     = self.jv_file

        if self.pvk_sample.performed_measurements is None:
            self.pvk_sample.performed_measurements = PerformedMeasurements()
        self.pvk_sample.performed_measurements.jv.append(summary)


m_package.__init_metainfo__()
