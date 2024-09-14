#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Functions for loading and treating CSV files downloaded from
CGU's Portal da Transparencia: http://www.portaltransparencia.gov.br/download-de-dados
Copyright (C) 2023  Henrique S. Xavier
Contact: hsxavier@gmail.com

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import pandas as pd
import requests
import zipfile
import io


def parse_ptbr_series(string_series):
    """
    Input: a Series of strings representing a float number in Brazilian currency format, e.g.: 1.573.345,98
    
    Returns a Series with the corresponding float number.
    """
    
    number_series = string_series.str.split().str.join('').str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
    return number_series


def download_zip_data(url, save_dir, verbose=False):
    """
    Download a zip file at `url` web address, unzip it and save it to `save_dir`.
    """
    # Initialize session:
    session = requests.session()
    session.mount('http://', requests.adapters.HTTPAdapter(max_retries=3))
    
    # Download data:
    if verbose:
        print('Downloading ' + url)
    response = session.get(url, timeout=900)
    
    # Unzip data if download is successful:
    if response.status_code == 200:
        if verbose:
            print('Unzipping data to ' + save_dir)
        z = zipfile.ZipFile(io.BytesIO(response.content))
        z.extractall(save_dir)
    else:
        if verbose:
            print('Failed with status code', response.status_code)
    
    return response.status_code    


def load_cgu_csv(filename, float_cols=[], date_cols=[], date_format='%d/%m/%Y', low_memory=False, all_str=False):
    """
    Load a CSV file downloaded from CGU's Portal da Transparência 
    into a DataFrame, taking advantage of their standard CSV format.

    Parameters
    ----------
    filename : str
        Path to the CSV file.
    float_cols : list of str
        List of columns containing float numbers to be properly parsed.
        Ignores non-existent columns.
    date_cols : list of str
        List of columns containing dates to be properly parsed. Ignores 
        non-existent columns.
    date_format : str
        Format used to parse the dates columns specified in `date_cols`,
        e.g. '%Y/%m'.
    low_memory : bool
        Whether to infer the data types based on a subset of data 
        or load the entire CSV file before deciding the data type.
    all_str : bool
        If True, set the data type for all columns to str before 
        parsing the columns in `float_cols` and `date_cols` (i.e. 
        these columns will have datetime and float types, respectively).

    Returns
    -------
    df : DataFrame
        The content of the CSV file with columns in `date_cols` and 
        `float_cols` parsed.
    """

    # Security checks:
    assert type(all_str) == bool

    # Set data type:
    if all_str:
        dtype=str
    else:
        dtype=None
        
    # Load data:
    df = pd.read_csv(filename, sep=';', encoding='latin-1', low_memory=low_memory, dtype=dtype)
    
    # Clean columns names:
    cols = df.columns
    clean_cols = [col.strip() for col in cols]
    df.rename(dict(zip(cols, clean_cols)), axis=1, inplace=True)
    
    # Parse float values:
    cols = df.columns
    for col in set(cols) & set(float_cols):
        df[col] = parse_ptbr_series(df[col])

    # Parse date values:
    cols = df.columns
    for col in set(cols) & set(date_cols):
        df[col] = pd.to_datetime(df[col], format=date_format)

    return df


def load_transferencias_file(filename, select_acoes=None):
    """
    Load data from a CSV file about transferências de recursos,
    downloaded from CGU's Portal da Transparência, and parse the
    relevant fields.
    
    If `select_acoes` is a list of str and not None, select 
    only ações with code is listed in `select_acoes`.
    """
    
    df = load_cgu_csv(filename, float_cols=['VALOR TRANSFERIDO']) 
    df['DATA'] = pd.to_datetime(df['ANO / MÊS'].astype(str) + '01', format='%Y%m%d')
    df['ANO']  = df['ANO / MÊS'].astype(str).str.slice(0, 4).astype(int)
    
    if select_acoes is not None:
        df = df.loc[df['AÇÃO'].isin(select_acoes)]
    
    return df


def load_despesas_file(filename, all_str=True, select_acoes=None):
    """
    Load Despesas data, obtained from Portal da Transparência.
    
    Parameters
    ----------
    filename : str
        CSV file containing the data, exactly as downloaded from
        Portal da Transparência.
    all_str : bool
        Whether to set all columns data types, apart from those 
        with date and float values, as strings.
    select_acoes : list of str, or None
        List of código de ações to select. Return eveything if 
        None.

    Returns
    -------
    df : DataFrame
    
    """
    
    # Hard-coded:
    date_cols  = ['Ano e mês do lançamento']
    float_cols = ['Valor Empenhado (R$)', 'Valor Liquidado (R$)', 'Valor Pago (R$)', 'Valor Restos a Pagar Inscritos (R$)', 'Valor Restos a Pagar Cancelado (R$)', 'Valor Restos a Pagar Pagos (R$)']
    
    # Load data:
    df = load_cgu_csv(filename, date_cols=date_cols, float_cols=float_cols, date_format='%Y/%m', all_str=all_str)
    
    # Filtra ações orçamentárias:
    if select_acoes is not None:
        df = df.loc[df['Código Ação'].isin(select_acoes)]

    return df
