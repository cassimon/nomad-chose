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


# class NewSchemaPackage(Schema):
#     name = Quantity(
#         type=str, a_eln=ELNAnnotation(component=ELNComponentEnum.StringEditQuantity)
#     )
#     message = Quantity(type=str)

#     def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
#         super().normalize(archive, logger)

#         logger.info('NewSchema.normalize', parameter=configuration.parameter)
#         self.message = f'Hello {self.name}!'




m_package = SchemaPackage()


class LabJVMeasurement(JVMeasurement, EntryData):
    """
    JV measurement entry.
    - pvk_sample: reference to sample → drives History/References tab
    - jv_file:    raw CSV file attached here (FileEditQuantity)
    - normalize() reads jv_file, parses it, fills self.results,
                  then copies scalar summary into sample.performed_measurements.jv[]
    No reference from sample back to this entry — cycle avoided.
    """
    m_def = Section(
        label='Lab JV Measurement',
        a_eln=dict(
            properties=dict(
                order=['name', 'pvk_sample', 'jv_file', 'datetime', 'operator']
            )
        ),
    )

    pvk_sample = Quantity(
        type=Reference(PerovskiteSolarCellSample.m_def),
        description='Sample this measurement was performed on.',
        a_eln=ELNAnnotation(component='ReferenceEditQuantity'),
    )
    jv_file = Quantity(
        type=str,
        description='Raw JV CSV file from the instrument.',
        a_eln=ELNAnnotation(
            component='FileEditQuantity',
            label='Raw JV file',
        ),
    )
    operator = Quantity(
        type=str,
        a_eln=ELNAnnotation(component='StringEditQuantity'),
    )

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        # ── Step 1: parse raw file if present ───────────────────────────────
        if self.jv_file and archive.m_context:
            try:
                raw_path = archive.m_context.raw_path(self.jv_file)
                jv_result = parse_jv_csv(raw_path, logger)
                if jv_result:
                    self.results = [jv_result]
                    # Also attach file path on the SolarCellJV result section
                    # so it shows up in the baseclasses file browser
                    self.results[0].data_file = self.jv_file
            except Exception as e:
                logger.warning(f'Could not parse JV file {self.jv_file}: {e}')

        # ── Step 2: copy scalar summary into sample.performed_measurements ───
        # One-directional: measurement → sample. No reference back.
        if self.pvk_sample is None:
            logger.warning(f'{self.__class__.__name__}: no pvk_sample set.')
            return

        if not self.results:
            return

        best = max(
            self.results,
            key=lambda r: r.efficiency if r.efficiency is not None else 0,
        )

        jv_summary = SolarCellJV()
        jv_summary.efficiency                    = best.efficiency
        jv_summary.open_circuit_voltage          = best.open_circuit_voltage
        jv_summary.short_circuit_current_density = best.short_circuit_current_density
        jv_summary.fill_factor                   = best.fill_factor
        jv_summary.light_intensity               = best.light_intensity
        jv_summary.data_file                     = self.jv_file

        if self.pvk_sample.performed_measurements is None:
            self.pvk_sample.performed_measurements = PerformedMeasurements()

        self.pvk_sample.performed_measurements.jv.append(jv_summary)
        logger.info(
            f'Appended JV summary (PCE={best.efficiency}) '
            f'into sample.performed_measurements.jv'
        )


m_package.__init_metainfo__()
