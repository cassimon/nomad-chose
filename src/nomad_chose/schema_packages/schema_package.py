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
from nomad.datamodel.data import ArchiveSection, EntryData
from nomad.datamodel.metainfo.annotations import ELNAnnotation
from nomad.metainfo import Quantity, Reference, Section, SubSection, SchemaPackage
import numpy as np
import os;
from baseclasses.solar_energy.jvmeasurement import JVMeasurement
from baseclasses.solar_energy.eqemeasurement import EQEMeasurement
from baseclasses.solar_energy.mpp_tracking import MPPTracking
from baseclasses.solar_energy.uvvismeasurement import UVvisData, UVvisMeasurement
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


class ChoseJVSettings(ArchiveSection):
    """The instrument's `[JV Settings]` and `[Cell Settings]` blocks.

    These have no home in `JVMeasurement`; keeping them in one flat section
    preserves them without inventing a class hierarchy. `scan_rate` and
    `voltage_step` are also read back out by the sample schema, which maps them
    onto the perovskite database's `jv.scan_speed` / `jv.scan_voltage_step`.
    """

    m_def = Section(label='JV Settings')

    v_min = Quantity(type=np.float64, unit='V', a_eln=dict(component='NumberEditQuantity'))
    v_max = Quantity(type=np.float64, unit='V', a_eln=dict(component='NumberEditQuantity'))
    voltage_step = Quantity(
        type=np.float64, unit='mV', a_eln=dict(component='NumberEditQuantity')
    )
    scan_rate = Quantity(
        type=np.float64, unit='mV/s', a_eln=dict(component='NumberEditQuantity')
    )
    scan_direction = Quantity(type=str, a_eln=dict(component='StringEditQuantity'))
    auto_detect_voc = Quantity(type=str, a_eln=dict(component='StringEditQuantity'))
    voltage_range = Quantity(type=str, a_eln=dict(component='StringEditQuantity'))
    current_range = Quantity(type=str, a_eln=dict(component='StringEditQuantity'))
    inverted = Quantity(type=str, a_eln=dict(component='StringEditQuantity'))
    auto_range = Quantity(type=str, a_eln=dict(component='StringEditQuantity'))

    typology = Quantity(type=str, a_eln=dict(component='StringEditQuantity'))
    number_of_cells = Quantity(type=np.float64, a_eln=dict(component='NumberEditQuantity'))
    wide_cell_area = Quantity(
        type=np.float64, unit='cm^2', a_eln=dict(component='NumberEditQuantity')
    )
    number_of_wide_cells = Quantity(
        type=np.float64, a_eln=dict(component='NumberEditQuantity')
    )

    def from_settings(self, settings):
        """Fill from the prefixed settings dict produced by `build_jv_dict`."""
        mapping = {
            'jv:vmin (v)': ('v_min', float),
            'jv:vmax (v)': ('v_max', float),
            'jv:voltage step (mv)': ('voltage_step', float),
            'jv:scan rate (mv/s)': ('scan_rate', float),
            'jv:scan direction': ('scan_direction', str),
            'jv:auto-detect voc': ('auto_detect_voc', str),
            'jv:voltage range (v)': ('voltage_range', str),
            'jv:current range (v)': ('current_range', str),
            'jv:inverted': ('inverted', str),
            'jv:auto-range': ('auto_range', str),
            'cell:tipology': ('typology', str),
            'cell:#cells': ('number_of_cells', float),
            'cell:w cell area (cm2)': ('wide_cell_area', float),
            'cell:#w cells': ('number_of_wide_cells', float),
        }
        for key, (attribute, cast) in mapping.items():
            raw = (settings or {}).get(key)
            if raw is None:
                continue
            try:
                setattr(self, attribute, cast(raw))
            except (TypeError, ValueError):
                continue
        return self


class LabJVMeasurement(JVMeasurement, EntryData):
    """
    JV measurement entry for the CHOSE lab instrument.

    """
    m_def = Section(
        label='CHOSE JV Measurement',
        a_eln=dict(
            properties=dict(
                order=['name', 'jv_file', 'datetime', 'operator', 'active_area', 'intensity']
            )
        ),
        # NOMAD-internal plotting of every parsed scan, as in nomad-hysprint.
        a_plot=[
            {
                'x': 'jv_curve/:/voltage',
                'y': 'jv_curve/:/current_density',
                'layout': {
                    'showlegend': True,
                    'yaxis': {'fixedrange': False},
                    'xaxis': {'fixedrange': False},
                },
            }
        ],
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

    settings = SubSection(section_def=ChoseJVSettings)

    def normalize(self, archive, logger):
        # Parse *before* super(), so that JVMeasurement.normalize sees a populated
        # `jv_curve` and can copy the best scan into
        # archive.results.properties.optoelectronic.solar_cell.
        if self.jv_file and archive is not None and archive.m_context:
            from nomad_chose.parsers.file_reading import apply_jv_dict, build_jv_dict

            try:
                text = _read_raw_text(archive, self.jv_file)
                jv_dict = build_jv_dict(
                    text,
                    self.jv_file,
                    intensity=self.intensity.magnitude if self.intensity is not None else None,
                    logger=logger,
                )
                if jv_dict:
                    apply_jv_dict(jv_dict, self.jv_file, self)
                    for curve in self.jv_curve:
                        curve.data_file = self.jv_file
                    if not self.operator and jv_dict.get('operator'):
                        self.operator = jv_dict['operator']
                    if not self.lab_id and jv_dict.get('device'):
                        self.lab_id = jv_dict['device']
                    if not self.description and jv_dict.get('note'):
                        self.description = jv_dict['note']
                    if jv_dict.get('settings'):
                        self.settings = ChoseJVSettings().from_settings(
                            jv_dict['settings']
                        )
            except Exception as e:
                logger.warning(f'LabJVMeasurement: could not parse {self.jv_file}: {e}')

        super().normalize(archive, logger)


class StabilityJVParameters(ArchiveSection):
    """The `Stability (Parameters)` table: JV figures of merit tracked over time.

    `MPPTracking` has nowhere to put a time series of *JV* parameters (it models
    the MPP track itself), so this is the one place the CHOSE data genuinely
    needs a new section.
    """

    m_def = Section(
        label='Stability JV Parameters',
        a_plot=[
            {
                'x': 'time',
                'y': ['efficiency_fw', 'efficiency_rv'],
                'layout': {
                    'showlegend': True,
                    'yaxis': {'fixedrange': False},
                    'xaxis': {'fixedrange': False},
                },
            }
        ],
    )

    time = Quantity(type=np.float64, shape=['*'], unit='hour')

    voc_fw = Quantity(type=np.float64, shape=['*'], unit='V')
    jsc_fw = Quantity(type=np.float64, shape=['*'], unit='mA/cm^2')
    v_mpp_fw = Quantity(type=np.float64, shape=['*'], unit='V')
    j_mpp_fw = Quantity(type=np.float64, shape=['*'], unit='mA/cm^2')
    p_mpp_fw = Quantity(type=np.float64, shape=['*'], unit='mW/cm^2')
    ff_fw = Quantity(type=np.float64, shape=['*'])
    efficiency_fw = Quantity(type=np.float64, shape=['*'])

    voc_rv = Quantity(type=np.float64, shape=['*'], unit='V')
    jsc_rv = Quantity(type=np.float64, shape=['*'], unit='mA/cm^2')
    v_mpp_rv = Quantity(type=np.float64, shape=['*'], unit='V')
    j_mpp_rv = Quantity(type=np.float64, shape=['*'], unit='mA/cm^2')
    p_mpp_rv = Quantity(type=np.float64, shape=['*'], unit='mW/cm^2')
    ff_rv = Quantity(type=np.float64, shape=['*'])
    efficiency_rv = Quantity(type=np.float64, shape=['*'])

    def from_parameters(self, parameters):
        if not parameters:
            return self
        time_hours = parameters.get('time_hours')
        if time_hours is not None and len(time_hours):
            self.time = time_hours
        for key, values in parameters.items():
            if key == 'time_hours' or values is None or not len(values):
                continue
            if self.m_def.all_properties.get(key) is None:
                continue
            setattr(self, key, values)
        return self


class LabStabilityMeasurement(MPPTracking, EntryData):
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
        a_plot=[
            {
                'x': 'time',
                'y': 'power_density',
                'layout': {
                    'showlegend': True,
                    'yaxis': {'fixedrange': False},
                    'xaxis': {'fixedrange': False},
                },
            }
        ],
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

    active_area = Quantity(
        type=np.float64,
        unit='cm^2',
        a_eln=dict(component='NumberEditQuantity', defaultDisplayUnit='cm^2'),
    )
    intensity = Quantity(
        type=np.float64,
        unit='mW/cm^2',
        a_eln=dict(component='NumberEditQuantity', defaultDisplayUnit='mW/cm^2'),
    )
    algorithm = Quantity(type=str, a_eln=ELNAnnotation(component='StringEditQuantity'))

    jv_parameters = SubSection(section_def=StabilityJVParameters)

    def normalize(self, archive, logger):
        if archive is not None and archive.m_context is not None:
            from baseclasses.solar_energy.mpp_tracking import MPPTrackingProperties

            from nomad_chose.parsers.file_reading import build_mppt_dict

            parameters_text = None
            tracking_text = None
            try:
                if self.stability_parameters_file:
                    parameters_text = _read_raw_text(
                        archive, self.stability_parameters_file
                    )
                if self.stability_tracking_file:
                    tracking_text = _read_raw_text(archive, self.stability_tracking_file)
            except Exception as e:
                logger.warning(
                    f'LabStabilityMeasurement: could not read stability files: {e}'
                )

            if parameters_text or tracking_text:
                data = build_mppt_dict(
                    parameters_text,
                    tracking_text,
                    intensity=self.intensity.magnitude
                    if self.intensity is not None
                    else None,
                    logger=logger,
                )

                # The native MPPTracking quantities: baseclasses reads these to
                # compute the T80/T95 figures of merit and to draw its figure.
                for attribute in (
                    'time',
                    'voltage',
                    'current_density',
                    'power_density',
                    'efficiency',
                    'active_area',
                    'intensity',
                    'algorithm',
                ):
                    value = data.get(attribute)
                    if value is not None:
                        setattr(self, attribute, value)

                if data.get('datetime'):
                    self.datetime = data['datetime']
                if not self.operator and data.get('operator'):
                    self.operator = data['operator']
                if not self.lab_id and data.get('device'):
                    self.lab_id = data['device']
                if not self.description and data.get('note'):
                    self.description = data['note']

                if data.get('total_time') or data.get('perturbation_voltage'):
                    self.properties = MPPTrackingProperties(
                        time=data.get('total_time'),
                        perturbation_voltage=data.get('perturbation_voltage'),
                        perturbation_delay=data.get('perturbation_delay'),
                    )

                if data.get('parameters'):
                    self.jv_parameters = StabilityJVParameters().from_parameters(
                        data['parameters']
                    )

        # Base normalize expects a real EntryArchive metadata.entry_name. Tests often
        # use a MagicMock archive that does not satisfy this requirement.
        metadata = getattr(archive, 'metadata', None)
        entry_name = getattr(metadata, 'entry_name', None)
        if not (isinstance(entry_name, str) and entry_name):
            return

        # MPPTracking.normalize derives the T80/T95 figures of merit with a
        # Savitzky-Golay filter whose window is len(power_density) // 5 at
        # polyorder 3 -- so a track shorter than 20 points makes scipy raise
        # ("polyorder must be less than window_length"). Hide the short track
        # from it: everything else on the entry still normalizes.
        points = 0 if self.power_density is None else len(self.power_density)
        if points and points // 5 <= 3:
            logger.info(
                f'LabStabilityMeasurement: {points} tracking points is too few for '
                'the stability figures of merit; skipping them.'
            )
            short_track = self.power_density
            self.power_density = None
            try:
                super().normalize(archive, logger)
            finally:
                self.power_density = short_track
            return

        super().normalize(archive, logger)


class ChoseEQESettings(ArchiveSection):
    """The IPCE export's `[Acquisition Settings]` and `[Device Settings]` blocks."""

    m_def = Section(label='EQE Settings')

    acquisition_time = Quantity(
        type=np.float64, unit='s', a_eln=dict(component='NumberEditQuantity')
    )
    averaging = Quantity(type=np.float64, a_eln=dict(component='NumberEditQuantity'))
    delay = Quantity(type=np.float64, unit='s', a_eln=dict(component='NumberEditQuantity'))
    led_level = Quantity(type=np.float64, a_eln=dict(component='NumberEditQuantity'))
    chopper_frequency = Quantity(
        type=np.float64, unit='Hz', a_eln=dict(component='NumberEditQuantity')
    )
    autorange = Quantity(type=str, a_eln=dict(component='StringEditQuantity'))
    current_range = Quantity(type=str, a_eln=dict(component='StringEditQuantity'))
    voltage_range = Quantity(type=str, a_eln=dict(component='StringEditQuantity'))


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
        a_plot=[
            {
                'x': 'eqe_data/:/photon_energy_array',
                'y': 'eqe_data/:/eqe_array',
                'layout': {
                    'showlegend': True,
                    'yaxis': {'fixedrange': False},
                    'xaxis': {'fixedrange': False},
                },
            }
        ],
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

    active_area = Quantity(
        type=np.float64,
        unit='cm^2',
        a_eln=dict(component='NumberEditQuantity', defaultDisplayUnit='cm^2'),
    )
    temperature = Quantity(
        type=np.float64,
        unit='celsius',
        a_eln=dict(component='NumberEditQuantity', defaultDisplayUnit='celsius'),
    )
    current_density_device = Quantity(type=np.float64, shape=['*'], unit='mA/cm^2')
    current_density_integrated = Quantity(type=np.float64, shape=['*'], unit='mA/cm^2')

    settings = SubSection(section_def=ChoseEQESettings)

    def normalize(self, archive, logger):
        if self.eqe_file and archive is not None and archive.m_context:
            from baseclasses.solar_energy import SolarCellEQECustom

            from nomad_chose.parsers.file_reading import build_eqe_dict

            try:
                text = _read_raw_text(archive, self.eqe_file)
                data = build_eqe_dict(text, logger)
                if data:
                    entry = SolarCellEQECustom(
                        photon_energy_array=data['photon_energy_array'],
                        raw_photon_energy_array=data['photon_energy_array'],
                        eqe_array=data['eqe_array'],
                        raw_eqe_array=data['eqe_array'],
                    )
                    if data.get('light_bias') is not None:
                        entry.light_bias = data['light_bias']
                    entry.eqe_data_file = self.eqe_file
                    # Runs the upstream analysis: bandgap_eqe, integrated_jsc,
                    # voc_rad, urbach_energy. Without this call none of it exists.
                    entry.normalize(archive, logger)
                    self.eqe_data = [entry]

                    for attribute in (
                        'active_area',
                        'temperature',
                        'current_density_device',
                        'current_density_integrated',
                    ):
                        value = data.get(attribute)
                        if value is not None:
                            setattr(self, attribute, value)
                    if data.get('datetime'):
                        self.datetime = data['datetime']
                    if not self.operator and data.get('operator'):
                        self.operator = data['operator']
                    if not self.lab_id and data.get('device'):
                        self.lab_id = data['device']

                    self.settings = ChoseEQESettings(
                        acquisition_time=data.get('integration_time'),
                        averaging=data.get('averaging'),
                        delay=data.get('delay'),
                        led_level=data.get('led_level'),
                        chopper_frequency=data.get('chopper_frequency'),
                        autorange=(data.get('settings') or {}).get('device:autorange'),
                        current_range=(data.get('settings') or {}).get(
                            'device:current range'
                        ),
                        voltage_range=(data.get('settings') or {}).get(
                            'device:voltage range'
                        ),
                    )
            except Exception as e:
                logger.warning(f'LabEQEMeasurement: could not parse {self.eqe_file}: {e}')

        super().normalize(archive, logger)


class LabUVvisMeasurement(UVvisMeasurement, EntryData):
    """A CHOSE UV-Vis transmittance spectrum of a film.

    Unlike JV/EQE/stability, this is a *film-level* measurement -- it describes a
    whole substrate, not a single pixel. The raw export carries only wavelength and
    transmittance, so the spectrum goes into the baseclass `measurements` (the
    UVvisData `intensity` field holds the transmittance in %).
    """

    m_def = Section(
        label='CHOSE UV-Vis Measurement',
        a_eln=dict(
            properties=dict(
                order=[
                    'name',
                    'uvvis_file',
                    'datetime',
                    'operator',
                ]
            )
        ),
        a_plot=[
            {
                'x': 'measurements/:/wavelength',
                'y': 'measurements/:/intensity',
                'layout': {
                    'showlegend': True,
                    'yaxis': {'fixedrange': False},
                    'xaxis': {'fixedrange': False},
                },
            }
        ],
    )

    uvvis_file = Quantity(
        type=str,
        description='Raw UV-Vis transmittance txt export from CHOSE instrument.',
        a_eln=ELNAnnotation(
            component='FileEditQuantity', label='Raw UV-Vis file (.txt)'
        ),
    )
    operator = Quantity(type=str, a_eln=ELNAnnotation(component='StringEditQuantity'))

    def normalize(self, archive, logger):
        if self.uvvis_file and archive is not None and archive.m_context:
            from nomad_chose.parsers.file_reading import build_uvvis_dict

            try:
                text = _read_raw_text(archive, self.uvvis_file)
                data = build_uvvis_dict(text, logger)
                if data:
                    entry = UVvisData(
                        name=data.get('name') or self.name,
                        wavelength=data['wavelength'],
                        intensity=data['transmittance'],
                    )
                    self.measurements = [entry]
                    if not self.name and data.get('name'):
                        self.name = data['name']
            except Exception as e:
                logger.warning(
                    f'LabUVvisMeasurement: could not parse {self.uvvis_file}: {e}'
                )

        super().normalize(archive, logger)


m_package.__init_metainfo__()
