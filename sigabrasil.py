#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Functions for loading and cleaning data from XLS files downloaded from SIGA Brasil
Painel Especialista: https://www12.senado.leg.br/orcamento/sigabrasil
Copyright (C) 2023  Henrique S. Xavier
Contact: hsxavier@gmail.com

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import pandas as pd


def convert_col(df, column, dtype):
    """
    Convert data type of `df` (DataFrame) `column` (str or int)
    to `dtype` (e.g. int, str, float) in place, if the 
    `column` exists in `df`. Otherwise, do nothing. 
    """
    if column in df.columns:
        df[column] = df[column].astype(dtype)

        
def convert_cols(df, columns, dtype):
    """
    Convert data type all `columns` (list of str or int) 
    of `df` (DataFrame) to `dtype` (type, e.g. int, str, 
    float) in place, if the the column exists in `df`. 
    Otherwise, skip the column. 
    """
    for col in columns:
        convert_col(df, col, dtype)

        
def load_sigabrasil_file(filename, drop_month_0=True):
    """
    Load SIGABrasil data stored in a XLS file and
    do some standard cleaning and data typing.
    
    The data is supposed to be obtained from 
    the Painel Especialista in SIGA Brasil 
    website.

    Parameters
    ----------
    drop_month_0 : bool
        Whether to remove lines referring to month 0, 
        in case the column 'Mês (Número) DES' is present.
        This column refers to the year total, if I am 
        not mistaken.

    Returns
    -------
    df : DataFrame
        The data from XLS file in a DataFrame.
    """
    
    # Carregando os dados:
    siga = pd.read_excel(filename)
    
    # Limpeza:
    
    # Remove linha de totalização:
    siga = siga.iloc[1:].reset_index(drop=True)
    cols = siga.columns
    
    # Converte tipos dos dados:
    int_cols = ['Ano', 'Mês (Número) DES', 'Subfunção (Cod) (Ajustado)', 'Função (Cod) DESP', 'Resultado Lei (Cod) DESP']
    convert_cols(siga, int_cols, int)
   
    # Remove year total if the data is per month:
    if drop_month_0 and ('Mês (Número) DES' in cols):
        siga = siga.loc[siga['Mês (Número) DES'] != 0]
        
    return siga


def load_sigabrasil(file_pattern, drop_month_0=True):
    """
    Load SIGABrasil data stored in multiple XLS
    files, following the glob pattern `file_pattern`
    (str), and do some standard cleaning and data typing.

    Parameters
    ----------
    drop_month_0 : bool
        Whether to remove lines referring to month 0, 
        in case the column 'Mês (Número) DES' is present.
        This column refers to the year total, if I am 
        not mistaken.

    Returns
    -------
    df : DataFrame
        The data from XLS file in a DataFrame.    
    """
    
    file_list = sorted(glob(file_pattern))

    siga = pd.concat([load_sigabrasil_file(filename, drop_month_0) for filename in file_list], ignore_index=True)

    return siga
