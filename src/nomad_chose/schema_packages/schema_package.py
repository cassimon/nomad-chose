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
import numpy as np

from baseclasses.solar_energy.jvmeasurement import JVMeasurement
from baseclasses.solar_energy.eqemeasurement import EQEMeasurement
from baseclasses.solar_energy.mpp_tracking import MPPTracking

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
        description='Raw JV file from the CHOSE instrument (.csv or .txt stability export).',
        a_eln=ELNAnnotation(
            component='FileEditQuantity',
            label='Raw JV file (.csv/.txt)',
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
            from nomad_chose.parsers.file_reading import parse_jv_file

            try:
                raw_path = archive.m_context.raw_path(self.jv_file)
                results = parse_jv_file(raw_path, logger)
                if results:
                    for result in results:
                        result.data_file = self.jv_file
                    self.results = results
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
        if self.pvk_sample.performed_measurements.jv is None:
            self.pvk_sample.performed_measurements.jv = []
        self.pvk_sample.performed_measurements.jv.append(summary)


class LabStabilityMeasurement(MPPTracking,EntryData):
    m_def = Section(
        label='CHOSE Stability Measurement',
        a_eln=dict(
            properties=dict(
                order=[
                    'name',
                    'pvk_sample',
                    'stability_parameters_file',
                    'stability_tracking_file',
                    'datetime',
                    'operator',
                ]
            )
        ),
    )

    pvk_sample = Quantity(
        type=Reference(PerovskiteSolarCellSample.m_def),
        description='The PerovskiteSolarCellSample this measurement belongs to.',
        a_eln=ELNAnnotation(component='ReferenceEditQuantity'),
    )
    stability_parameters_file = Quantity(
        type=str,
        a_eln=ELNAnnotation(component='FileEditQuantity', label='Stability Parameters file'),
    )
    stability_tracking_file = Quantity(
        type=str,
        a_eln=ELNAnnotation(component='FileEditQuantity', label='Stability Tracking file'),
    )
    operator = Quantity(type=str, a_eln=ELNAnnotation(component='StringEditQuantity'))

    time_hours = Quantity(type=np.float64, shape=['*'])
    efficiency_fw = Quantity(type=np.float64, shape=['*'])
    efficiency_rv = Quantity(type=np.float64, shape=['*'])
    tracking_time_hours = Quantity(type=np.float64, shape=['*'])
    tracking_voltage = Quantity(type=np.float64, shape=['*'])
    tracking_current_density = Quantity(type=np.float64, shape=['*'])
    tracking_power = Quantity(type=np.float64, shape=['*'])

    def normalize(self, archive, logger):
        super().normalize(archive, logger)
        if archive is None or archive.m_context is None:
            return

        from nomad_chose.parsers.file_reading import parse_stability_pair

        parameters_path = (
            archive.m_context.raw_path(self.stability_parameters_file)
            if self.stability_parameters_file
            else None
        )
        tracking_path = (
            archive.m_context.raw_path(self.stability_tracking_file)
            if self.stability_tracking_file
            else None
        )

        parsed = parse_stability_pair(parameters_path, tracking_path)

        if parsed.parameters:
            self.time_hours = parsed.parameters.get('time_hours')
            self.efficiency_fw = parsed.parameters.get('efficiency_fw')
            self.efficiency_rv = parsed.parameters.get('efficiency_rv')

        if parsed.tracking:
            self.tracking_time_hours = parsed.tracking.get('time_hours')
            self.tracking_voltage = parsed.tracking.get('voltage')
            self.tracking_current_density = parsed.tracking.get('current_density')
            self.tracking_power = parsed.tracking.get('power')


class LabEQEMeasurement(EQEMeasurement, EntryData):
    m_def = Section(
        label='CHOSE EQE Measurement',
        a_eln=dict(
            properties=dict(
                order=[
                    'name',
                    'pvk_sample',
                    'eqe_file',
                    'datetime',
                    'operator',
                ]
            )
        ),
    )

    pvk_sample = Quantity(
        type=Reference(PerovskiteSolarCellSample.m_def),
        description='The PerovskiteSolarCellSample this measurement belongs to.',
        a_eln=ELNAnnotation(component='ReferenceEditQuantity'),
    )
    eqe_file = Quantity(
        type=str,
        description='Raw IPCE/EQE txt export from CHOSE instrument.',
        a_eln=ELNAnnotation(component='FileEditQuantity', label='Raw EQE file (.txt)'),
    )
    operator = Quantity(type=str, a_eln=ELNAnnotation(component='StringEditQuantity'))

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        if self.eqe_file and archive is not None and archive.m_context:
            from nomad_chose.parsers.file_reading import parse_ipce_file

            try:
                raw_path = archive.m_context.raw_path(self.eqe_file)
                result = parse_ipce_file(raw_path, logger)
                if result is not None:
                    result.data_file = self.eqe_file
                    self.eqe_data = [result]
            except Exception as e:
                logger.warning(f'LabEQEMeasurement: could not parse {self.eqe_file}: {e}')


m_package.__init_metainfo__()
