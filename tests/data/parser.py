from typing import (
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import (
        EntryArchive
    )
    from structlog.stdlib import (
        BoundLogger,
    )

from nomad.config import config
from nomad.datamodel.metainfo.workflow import Workflow
from nomad.parsing.parser import MatchingParser
import numpy as np
#importa ArchiveSection
from nomad.metainfo import Section, Quantity  # Added import
from nomad.datamodel.data import ArchiveSection # Added import, replaces commented out #importa ArchiveSection
from nomad.datamodel.data import EntryData
from baseclasses.solar_energy import JVMeasurement
from baseclasses.solar_energy import EQEMeasurement, SolarCellEQECustom,  MPPTrackingHsprintCustom
from baseclasses.helper.add_solar_cell import add_band_gap
import os;
from baseclasses.helper.utilities import set_sample_reference, create_archive, get_entry_id_from_file_name, get_reference
import datetime;
from nomad.datamodel.metainfo.basesections import (
    Entity,
)
from nomad.datamodel.metainfo.annotations import (
    ELNAnnotation,
)
from baseclasses.helper.utilities import get_encoding
from nomad.datamodel.metainfo.plot import PlotlyFigure, PlotSection
from nomad.units import ureg

class RawFileUNITOV(EntryData):
    processed_archive = Quantity(
        type=Entity,
        a_eln=ELNAnnotation(
            component='ReferenceEditQuantity',
        )
    )


class UNITOV_JVmeasurement(JVMeasurement, RawFileUNITOV):
    m_def = Section(
        a_eln=dict(
            hide=[
                'lab_id', 'solution',
                'users',
                'author',
                'certified_values',
                'certification_institute',
                'end_time', 'steps', 'instruments', 'results',
            ],
            properties=dict(
                order=[
                    "name",
                    "data_file",
                    "active_area",
                    "intensity",
                    "integration_time",
                    "settling_time",
                    "averaging",
                    "compliance",
                    "samples"])),
        a_plot=[
            {
                'x': 'jv_curve/:/voltage',
                'y': 'jv_curve/:/current_density',
                'layout': {
                    "showlegend": True,
                    'yaxis': {
                        "fixedrange": False},
                    'xaxis': {
                        "fixedrange": False}},
            }])

    def normalize(self, archive, logger):
        print("starting normalize JV with datafile: ",self.data_file)
        if self.data_file:
            # todo detect file format

            with archive.m_context.raw_file(self.data_file, "br") as f:
                encoding = get_encoding(f)

            with archive.m_context.raw_file(self.data_file, "tr", encoding=encoding) as f:
                from nomad_test_parser.parsers.file_reading import read_file_jv_data
                from baseclasses.helper.archive_builder.jv_archive import get_jv_archive

                jv_dict, location = read_file_jv_data(f)

                self.location = location
                get_jv_archive(jv_dict, self.data_file, self)

            print("=======================================WRITTEN DATA IN ARCHIVE")

        super(UNITOV_JVmeasurement,
              self).normalize(archive, logger)


class UNITOV_MPPTracking_Measurement(MPPTrackingHsprintCustom, PlotSection, RawFileUNITOV):
    m_def = Section(
        a_eln=dict(
            hide=[
                'lab_id',
                'solution',
                'users',
                'author',
                'end_time',
                'steps',
                'instruments',
                'results',
                'location',
                'figures',
            ],
            properties=dict(
                order=[
                    'name',
                    'data_file',
                    'active_area',
                    'intensity',
                    'integration_time',
                    'settling_time',
                    'averaging',
                    'compliance',
                    'samples',
                ]
            ),
        )
    )

    def normalize(self, archive, logger):
        from baseclasses.helper.archive_builder.mpp_hysprint_archive import get_mpp_hysprint_samples
        from baseclasses.helper.utilities import rewrite_json

        from nomad_test_parser.parsers.file_reading import read_mppt_file

        if self.data_file and self.load_data_from_file:
            self.load_data_from_file = False

            #rewrite_json(['data', 'load_data_from_file'], archive, False)

            # from baseclasses.helper.utilities import get_encoding
            # with archive.m_context.raw_file(self.data_file, "br") as f:
            #     encoding = get_encoding(f)

            with archive.m_context.raw_file(self.data_file, 'tr', encoding='ascii') as f:
                if os.path.splitext(f.name)[-1] != '.csv':
                    return

                data = read_mppt_file(f.read())  # , encoding)

            self.samples = get_mpp_hysprint_samples(self, data)

        import pandas as pd
        import plotly.express as px

        column_names = ['Time [hr]', 'Efficiency [%]', 'P']
        self.figures = []
        if self.averages:
            fig = px.scatter()
            df = pd.DataFrame(columns=column_names)
            for avg in self.averages:
                df1 = pd.DataFrame(columns=column_names)
                df1[column_names[0]] = avg.time * ureg('hr')
                df1[column_names[1]] = avg.efficiency
                df1[column_names[2]] = avg.name
                df = pd.concat([df, df1])

            fig = px.scatter(
                df,
                x=column_names[0],
                y=column_names[1],
                color=column_names[2],
                symbol=column_names[2],
                title='Averages',
            )
            fig.update_traces(marker=dict(size=4))
            fig.update_layout(
                showlegend=True,
                xaxis=dict(fixedrange=False),
                yaxis=dict(fixedrange=False),
            )
            self.figures.append(PlotlyFigure(label='Averages', index=0, figure=fig.to_plotly_json()))

        if self.best_pixels:
            df = pd.DataFrame(columns=column_names)
            for bp in self.best_pixels:
                df1 = pd.DataFrame(columns=column_names)
                df1[column_names[0]] = bp.time * ureg('hr')
                df1[column_names[1]] = bp.efficiency
                df1[column_names[2]] = bp.name
                df = pd.concat([df, df1])

            fig = px.scatter(
                df,
                x=column_names[0],
                y=column_names[1],
                color=column_names[2],
                symbol=column_names[2],
                title='Best Pixels',
            )
            fig.update_traces(marker=dict(size=4))
            fig.update_layout(
                showlegend=True,
                xaxis=dict(fixedrange=False),
                yaxis=dict(fixedrange=False),
            )
            self.figures.append(PlotlyFigure(label='Best Pixel', index=1, figure=fig.to_plotly_json()))

        super().normalize(archive, logger)

class UNITOV_EQEmeasurement(EQEMeasurement, RawFileUNITOV):
    m_def = Section(
        a_eln=dict(
            hide=[
                'lab_id',
                'solution',
                'users',
                'location',
                'end_time',
                'steps',
                'instruments',
                'results',
                'data',
                'header_lines',
            ],
            properties=dict(order=['name', 'data_file', 'samples']),
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

    def normalize(self, archive, logger):
        from nomad_test_parser.parsers.file_reading import read_file_eqe

        #if not self.samples and self.data_file:
        #    search_id = self.data_file.split('.')[0]
        #    set_sample_reference(archive, self, search_id, upload_id=archive.metadata.upload_id)

        if self.data_file:
            with archive.m_context.raw_file(self.data_file, 'br') as f:
                encoding = get_encoding(f)

            with archive.m_context.raw_file(self.data_file, 'tr', encoding=encoding) as f:
                eqe_dict,UPLOAD_FLAG = read_file_eqe(f,header_lines=24)


            entry = SolarCellEQECustom(
                photon_energy_array=eqe_dict.get('photon_energy'),
                raw_photon_energy_array=eqe_dict.get('photon_energy_raw'),
                eqe_array=eqe_dict.get('intensity'),
                raw_eqe_array=eqe_dict.get('intensty_raw'),
            )
            entry.normalize(archive, logger)
            eqe_data = list();
            eqe_data.append(entry)

            self.eqe_data = eqe_data

        # if eqe_data:
        #     band_gaps = np.array([d.bandgap_eqe.magnitude for d in eqe_data])

        #     add_band_gap(archive, band_gaps[np.isfinite(band_gaps)].mean())

        super().normalize(archive, logger)



configuration = config.get_plugin_entry_point(
    'nomad_test_parser.parsers:parser_entry_point'
)

class SimpleOutput(ArchiveSection):
    """
    A simple section to store basic output from this parser.
    """
    m_def = Section(label='Simple Parser Output') # Defines the section metadata

    message = Quantity(
        type=str,
        description='A simple message generated by the parser.'
    )
    example_value = Quantity(
        type=np.float64,
        description='An example numerical value processed by the parser.'
    )
    parsed_mainfile_path = Quantity(
        type=str,
        description='The path of the mainfile that was parsed.'
    )





class NewParser(MatchingParser):
    def parse(
        self,
        mainfile: str,
        archive: 'EntryArchive',
        logger: 'BoundLogger',
        child_archives: dict[str, 'EntryArchive'] = None,
    ) -> None:
        logger.info('JVParser.parse', parameter=configuration.parameter)
        print("PIPPO ==========================================")
        print("PIPPO ==========================================")
        print("PIPPO ==========================================")
        print("PIPPO ==========================================")

        # 1. Create an instance of your custom section
        my_simple_output_data = SimpleOutput()

        # 2. Populate the section with some data
        #    (In a real parser, this data would come from parsing the 'mainfile')
        my_simple_output_data.message = 'This is a test message from NewParser.'
        my_simple_output_data.example_value = 123.456
        my_simple_output_data.parsed_mainfile_path = mainfile

        # 3. Assign your custom section to archive.data
        #    archive.data is the typical place for the primary data extracted by the parser.
        archive.data = my_simple_output_data


class EQEParser(MatchingParser):
    def parse(
        self,
        mainfile: str,
        archive: 'EntryArchive',
        logger: 'BoundLogger',
        child_archives: dict[str, 'EntryArchive'] = None,
    ) -> None:
        logger.info('EQEParser.parse', parameter=configuration.parameter)
        print("Parse EQE ==========================================")

        # 1. Create an instance of your custom section
        #entry = UNITOV_JVmeasurement()
        notes = ''

        basename = os.path.basename(mainfile)
        file_name = f'{basename}.archive.json'
        archive.data = UNITOV_EQEmeasurement()
        archive.data.message = 'This is a test EQE measurement parsing.'
        archive.data.datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        archive.data.data_file = basename;
        archive.data.m_to_dict(with_root_def=True)
        create_archive(archive.data, archive, file_name, overwrite=True)

class MPPTParser(MatchingParser):
    def parse(
        self,
        mainfile: str,
        archive: 'EntryArchive',
        logger: 'BoundLogger',
        child_archives: dict[str, 'EntryArchive'] = None,
    ) -> None:
        logger.info('MPPTParser.parse', parameter=configuration.parameter)
        print("Parse MPPT ==========================================")

        # 1. Create an instance of your custom section
        #entry = UNITOV_JVmeasurement()
        notes = ''

        file_name = f'{os.path.basename(mainfile)}.archive.json'

        archive.data = UNITOV_MPPTracking_Measurement()
        archive.data.message = 'This is a test MPPT measurement parsing.'
        archive.data.datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        archive.data.data_file = os.path.basename(mainfile)
        archive.data.m_to_dict(with_root_def=True)
        create_archive(archive.data, archive, file_name, overwrite=True)


class JVParser(MatchingParser):
    def parse(
        self,
        mainfile: str,
        archive: 'EntryArchive',
        logger: 'BoundLogger',
        child_archives: dict[str, 'EntryArchive'] = None,
    ) -> None:
        logger.info('NewParser.parse', parameter=configuration.parameter)
        print("Parse JV curve ==========================================")

        # 1. Create an instance of your custom section
        #entry = UNITOV_JVmeasurement()
        notes = ''

        # 2. Populate the section with some data
        #    (In a real parser, this data would come from parsing the 'mainfile')

        #archive.metadata.entry_name = os.path.basename(mainfile)

        #mainfile_split = mainfile.split('.');

        #if not mainfile_split[-1] in ["nk"]:
        #    search_id = mainfile_split[0]
        #    set_sample_reference(archive, entry, search_id)

        #    entry.name = f"{search_id} {notes}"
        #    entry.description = f"Notes from file name: {notes}"

        #if not measurment_type in ["uvvis", "sem", "SEM"]:
        #    entry.data_file = os.path.basename(mainfile)



        file_name = f'{os.path.basename(mainfile)}.archive.json'
        #eid = get_entry_id_from_file_name(file_name, archive)

        #archive.data = UNITOV_JVmeasurement(processed_archive=get_reference(archive.metadata.upload_id, eid))
        archive.data = UNITOV_JVmeasurement()
        basename = os.path.basename(mainfile)
        archive.data.message = 'This is a test JV measurement parsing.'
        archive.data.datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        archive.data.data_file = basename;
        archive.data.m_to_dict(with_root_def=True)
        create_archive(archive.data, archive, file_name, overwrite=True)


        # 3. Assign your custom section to archive.data
        #    archive.data is the typical place for the primary data extracted by the parser.
        #archive.data = my_simple_output_data
