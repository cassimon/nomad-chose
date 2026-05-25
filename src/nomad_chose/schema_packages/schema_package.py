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
import os;
from baseclasses.solar_energy.jvmeasurement import JVMeasurement
from baseclasses.solar_energy.eqemeasurement import EQEMeasurement
from baseclasses.solar_energy.mpp_tracking import MPPTracking
from baseclasses.helper.utilities import get_encoding

configuration = config.get_plugin_entry_point(
    'nomad_chose.schema_packages:schema_package_entry_point'
)


def _read_raw_text(archive, raw_name: str) -> str:
    """Read a NOMAD raw file using raw_file, with a raw_path fallback for tests."""
    context = archive.m_context

    # Preferred NOMAD API.
    try:
        with context.raw_file(raw_name, 'br') as f:
            encoding = get_encoding(f)
        with context.raw_file(raw_name, 'tr', encoding=encoding) as f:
            text = f.read()
            if isinstance(text, str):
                return text
            raise TypeError('raw_file did not return text content')
    except Exception:
        pass

    # Fallback for test mocks that only provide raw_path.
    raw_path_attr = getattr(context, 'raw_path', None)
    if raw_path_attr is None:
        raise FileNotFoundError(f'Could not resolve raw file {raw_name}')

    path = None
    if callable(raw_path_attr):
        try:
            path = raw_path_attr(raw_name)
        except TypeError:
            path = os.path.join(raw_path_attr(), raw_name)
    else:
        path = os.path.join(str(raw_path_attr), raw_name)

    with open(path, 'rb') as f:
        encoding = get_encoding(f)
    with open(path, 'r', encoding=encoding, errors='replace') as f:
        return f.read()


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
                order=['name', 'jv_file', 'datetime', 'operator']
            )
        ),
    )

    # pvk_sample = Quantity(
    #     type=Reference(PerovskiteSolarCellSample.m_def),
    #     description='The PerovskiteSolarCellSample this measurement belongs to.',
    #     a_eln=ELNAnnotation(component='ReferenceEditQuantity'),
    # )

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
            from nomad_chose.parsers.file_reading import parse_jv_file_from_text

            try:
                text = _read_raw_text(archive, self.jv_file)
                results = parse_jv_file_from_text(text, self.jv_file, logger)
                if results:
                    for result in results:
                        result.data_file = self.jv_file
                    self.results = results
            except Exception as e:
                logger.warning(f'LabJVMeasurement: could not parse {self.jv_file}: {e}')

        if not self.results:
            return


class LabStabilityMeasurement(MPPTracking,EntryData):
    m_def = Section(
        label='CHOSE Stability Measurement',
        a_eln=dict(
            properties=dict(
                order=[
                    'name',
                    'stability_parameters_file',
                    'stability_tracking_file',
                    'datetime',
                    'operator',
                ]
            )
        ),
    )

    # pvk_sample = Quantity(
    #     type=Reference(PerovskiteSolarCellSample.m_def),
    #     description='The PerovskiteSolarCellSample this measurement belongs to.',
    #     a_eln=ELNAnnotation(component='ReferenceEditQuantity'),
    # )
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
        # Base normalize expects a real EntryArchive metadata.entry_name. Tests often
        # use a MagicMock archive that does not satisfy this requirement.
        metadata = getattr(archive, 'metadata', None)
        entry_name = getattr(metadata, 'entry_name', None)
        if isinstance(entry_name, str) and entry_name:
            super().normalize(archive, logger)
        if archive is None or archive.m_context is None:
            return

        from nomad_chose.parsers.file_reading import parse_stability_pair_from_text

        parameters_text = None
        tracking_text = None
        try:
            if self.stability_parameters_file:
                parameters_text = _read_raw_text(archive, self.stability_parameters_file)

            if self.stability_tracking_file:
                tracking_text = _read_raw_text(archive, self.stability_tracking_file)
        except Exception as e:
            logger.warning(f'LabStabilityMeasurement: could not read stability files: {e}')

        parsed = parse_stability_pair_from_text(parameters_text, tracking_text)

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
                    'eqe_file',
                    'datetime',
                    'operator',
                ]
            )
        ),
    )

    #  pvk_sample = Quantity(
    #     type=Reference(PerovskiteSolarCellSample.m_def),
    #     description='The PerovskiteSolarCellSample this measurement belongs to.',
    #     a_eln=ELNAnnotation(component='ReferenceEditQuantity'),
    # )
    eqe_file = Quantity(
        type=str,
        description='Raw IPCE/EQE txt export from CHOSE instrument.',
        a_eln=ELNAnnotation(component='FileEditQuantity', label='Raw EQE file (.txt)'),
    )
    operator = Quantity(type=str, a_eln=ELNAnnotation(component='StringEditQuantity'))

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        if self.eqe_file and archive is not None and archive.m_context:
            from nomad_chose.parsers.file_reading import parse_ipce_file_from_text

            try:
                text = _read_raw_text(archive, self.eqe_file)
                result = parse_ipce_file_from_text(text, self.eqe_file, logger)
                if result is not None:
                    result.data_file = self.eqe_file
                    self.eqe_data = [result]
            except Exception as e:
                logger.warning(f'LabEQEMeasurement: could not parse {self.eqe_file}: {e}')


m_package.__init_metainfo__()
