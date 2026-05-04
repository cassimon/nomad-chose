#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD. See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pandas as pd
import numpy as np
import ast
from io import StringIO
import io;


UPDLOADED_FLAG = 'UNITOV';

# Constants
temperature = 300  # in [°K]
q = 1.602176462e-19  # % [As], elementary charge
h_Js = 6.62606876e-34  # % [Js], Planck's constant
k = 1.38064852e-23  # % [(m^2)kg(s^-2)(K^-1)], Boltzmann constant
T = temperature
VT = (k * T) / q  # % [V], 25.8mV thermal voltage at 300K
c = 299792458  # % [m/s], speed of light c_0
hc_eVnm = h_Js * c / q * 1e9  # % [eV nm]  Planck's constant for energy to wavelength conversion


def get_value(val):
    try:
        return float(val)
    except:
        return None


def read_mppt_file(filedata):
    filedata = filedata.replace("²", "^2")

    df = pd.read_csv(
        StringIO(filedata),
        skiprows=0,
        nrows=41,
        header=None,
        sep='\t',
        index_col=0,
        engine='python',
        encoding='unicode_escape').T

    df_curve = pd.read_csv(
        StringIO(filedata),
        skiprows=42,
        sep='\t',
        encoding='unicode_escape',
        engine='python')
    df_curve = df_curve.dropna(how='any', axis=0)

    print(df.T,df_curve)

    mppt_dict = {}
    mppt_dict['total_time'] = get_value(df['Test duration (hours)'].iloc[0]*60*60)
    mppt_dict['step_size'] = get_value(df['JV interval (min)'].iloc[0]*60)
    mppt_dict['time_per_track'] = get_value(df['track delay (s)'].iloc[0])
    mppt_dict['active_area'] = get_value(df['Cell Area (cm2)'].iloc[0])
    #mppt_dict['voltage'] = get_value(df.iloc[4, 0])

    mppt_dict['time_data'] = np.array(df_curve["Time (hours)"], dtype=np.float64)
    mppt_dict['voltage_data'] = np.array(df_curve["V (V)"], dtype=np.float64)
    mppt_dict['current_density_data'] = np.array(df_curve["J (mAcm-2)"], dtype=np.float64)
    mppt_dict['power_data'] = np.array(df_curve["P (mWcm-2)"], dtype=np.float64)

    return mppt_dict,UPDLOADED_FLAG


def interpolate_eqe(photon_energy_raw, eqe_raw):
    photon_energy_interpolated = np.linspace(min(photon_energy_raw), max(photon_energy_raw), 1000, endpoint=True)
    eqe_interpolated = np.interp(photon_energy_interpolated, photon_energy_raw, eqe_raw)

    return photon_energy_interpolated, eqe_interpolated


def arrange_eqe_columns(df):
    """
    Gets a df with columns of the file and returns a `photon_energy_raw` array
    and `eqe_raw` array with values of the photon energy values in *eV* and
    the eqe (values between 0 and 1) respectively.
    It finds if the eqe data comes in nm or eV and converts it to eV.

    Returns:
        photon_energy_raw: array of photon energy values in eV
        eqe_raw: array of eqe values
    """

    x = np.array(df['Wavelength (nm)']) # for files from the unitov
    y = np.array(df['IPCE (%)'])


    if any(x > 10):  # check if energy (eV) or wavelength (nm)
        x = hc_eVnm / x
    if any(y > 10):  # check if EQE is given in (%), if so it's translated to abs. numbers
        y = y / 100

    # bring both arrays into correct order (i.e. w.r.t eV increasing) if one started with e.g. wavelength in increasing order e.g. 300nm, 305nm,...
    if x[1] - x[2] > 0:
        x = np.flip(x)
        y = np.flip(y)

    photon_energy_raw = x
    eqe_raw = y
    return photon_energy_raw, eqe_raw


def read_file_eqe(filedata, header_lines=None):
    """
    Reads the file and returns the columns in a pandas DataFrame `df`.
    :return: df
    :rtype: pandas.DataFrame
    """

    if header_lines is None:
        header_lines = 0
    if header_lines == 0:  # in case you have a header
        try:
            df = pd.read_csv(StringIO(filedata.read()), header=None, sep='\t',)
            if len(df.columns) < 2:
                raise IndexError
        except IndexError:
            df = pd.read_csv(StringIO(filedata.read()), header=None)
    else:
        try:
            # header_lines - 1 assumes last header line is column names
            df = pd.read_csv(StringIO(filedata.read()), header=int(header_lines - 1), sep='\t')
            if len(df.columns) < 2:
                raise IndexError
        except IndexError:
            try:  # wrong separator?
                df = pd.read_csv(StringIO(filedata.read()), header=int(header_lines - 1))
                if len(df.columns) < 2:
                    raise IndexError
            except IndexError:
                try:  # separator was right, but last header_line is not actually column names?
                    df = pd.read_csv(StringIO(filedata.read()), header=int(header_lines), sep='\t')
                    if len(df.columns) < 2:
                        raise IndexError
                except IndexError:
                    # Last guess: separator was wrong AND last header_line is not actually column names?
                    df = pd.read_csv(StringIO(filedata.read()), header=int(header_lines))
                    if len(df.columns) < 2:
                        raise IndexError
    #print(df)

    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    #print(df)
    photon_energy_raw, eqe_raw = arrange_eqe_columns(df)
    photon_energy, intensity = interpolate_eqe(photon_energy_raw, eqe_raw)
    return {'photon_energy_raw':photon_energy_raw, 'eqe_raw':eqe_raw, 'photon_energy':photon_energy, 'intensity':intensity},UPDLOADED_FLAG


def read_file_jv_data_stab(filedata):
    # Block to clean up some bad characters found in the file which gives
    # trouble reading.

    #filedata = filedata.replace("²", "^2")

    df = pd.read_csv(
        StringIO(filedata),
        skiprows=43,
        nrows=2,
        sep='\t',
        index_col=0,
        engine='python',
        encoding='unicode_escape')

    df_header = pd.read_csv(StringIO(filedata),
                            skiprows=2,
                            nrows=39,
                            header=None,
                            sep='\t',
                            index_col=0,
                            encoding='unicode_escape',
                            engine='python')

    nlines = filedata.count('\n')

    df_curves = pd.read_csv(StringIO(filedata),
                            skiprows=47,
                            nrows=nlines-47-2,
                            sep='\t',
                            encoding='unicode_escape',
                            engine='python')


    df_curves = df_curves.dropna(how='all', axis=1)

    df_header.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

    number_of_curves = len(df_curves.columns)-1

    print(df_header)

    jv_dict = {}
    jv_dict['active_area'] = df_header.iloc[1, 17]
    jv_dict['intensity'] = df_header.iloc[1, 1]
    jv_dict['integration_time'] = df_header.iloc[2, 1]
    jv_dict['settling_time'] = df_header.iloc[3, 1]
    jv_dict['averaging'] = df_header.iloc[4, 1]
    jv_dict['compliance'] = df_header.iloc[5, 1]

    jv_dict['J_sc'] = list(abs(df.iloc[0]))[:number_of_curves]
    jv_dict['V_oc'] = list(df.iloc[1])[:number_of_curves]
    jv_dict['Fill_factor'] = list(df.iloc[2])[:number_of_curves]
    jv_dict['Efficiency'] = list(df.iloc[3])[:number_of_curves]
    jv_dict['P_MPP'] = list(df.iloc[4])[:number_of_curves]
    jv_dict['J_MPP'] = list(abs(df.iloc[5]))[:number_of_curves]
    jv_dict['U_MPP'] = list(df.iloc[6])[:number_of_curves]
    jv_dict['R_ser'] = list(df.iloc[7])[:number_of_curves]
    jv_dict['R_par'] = list(df.iloc[8])[:number_of_curves]
    jv_dict['jv_curve'] = []


    for column in range(1, len(df_curves.columns)):
        jv_dict['jv_curve'].append({'name': df_curves.columns[column],
                                    'voltage': df_curves[df_curves.columns[0]].values,
                                    'current_density': df_curves[df_curves.columns[column]].values})



    return jv_dict,UPDLOADED_FLAG


def read_file_jv_data(filedata):
    # Block to clean up some bad characters found in the file which gives
    # trouble reading.

    #filedata = filedata.replace("²", "^2")
    with filedata as file:
        content = file.read();
        df = pd.read_csv(
            StringIO(content),
            skiprows=42,
            nrows=3,
            sep='\t',
            index_col=0,
            engine='python',
            encoding='unicode_escape')

        df_header = pd.read_csv(StringIO(content),
                                skiprows=2,
                                nrows=39,
                                header=None,
                                sep='\t',
                                index_col=0,
                                encoding='unicode_escape',
                                engine='python').T


        nlines = content.count('\n')

        df_curves = pd.read_csv(StringIO(content),
                                skiprows=47,
                                nrows=nlines-47-2,
                                sep='\t',
                                encoding='unicode_escape',
                                engine='python')


    df_curves = df_curves.dropna(how='all', axis=1)

    df_header.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

    #number_of_curves = len(df_curves.columns)-1
    number_of_curves = 2

    print(df_header.T)

    jv_dict = {}
    jv_dict['active_area'] = float(df_header['Cell Area (cm2)'].iloc[0])
    jv_dict['intensity'] = 100


    #jv_dict['integration_time'] = df_header.iloc[2, 1]
    #jv_dict['settling_time'] = df_header.iloc[3, 1]
    #jv_dict['averaging'] = df_header.iloc[4, 1]
    #jv_dict['compliance'] = df_header.iloc[5, 1]

    df = df.drop([np.nan]).astype(float)

    #print(df)

    jv_dict['J_sc'] = list(abs(df['Jsc']))[:number_of_curves]
    jv_dict['V_oc'] = list(abs(df['Voc']))[:number_of_curves]

    #print(jv_dict['V_oc'])

    jv_dict['Fill_factor'] = list(df['FF'])[:number_of_curves]
    jv_dict['Efficiency'] = list(df['Eff'])[:number_of_curves]
    jv_dict['P_MPP'] = list(df['P_MPP'])[:number_of_curves]
    jv_dict['J_MPP'] = list(abs(df['J_MPP']))[:number_of_curves]
    jv_dict['U_MPP'] = list(df['V_MPP'])[:number_of_curves]
    jv_dict['R_ser'] = list(df['Rs'])[:number_of_curves]
    jv_dict['R_par'] = list(df['R//'])[:number_of_curves]
    jv_dict['jv_curve'] = []


    for n in range(0, len(df_curves.columns),2):

        jv_dict['jv_curve'].append({'name': "Scan " + str(n+1) ,
                                    'voltage': df_curves[df_curves.columns[n]].values,
                                    'current_density': df_curves[df_curves.columns[n+1]].values})



    return jv_dict,UPDLOADED_FLAG





#import os;
#abs_path = os.path.dirname(os.path.abspath(__file__))+'\\';
#file = io.open(abs_path+r"..\..\..\..\tests\data\example_measurement\001_JV.txt", "r",encoding='windows-1252')
#print(get_jv_data_unitov(file.read()))