#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Functions for interacting with Google Cloud Platform (Storage and BigQuery)
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

import os
import pandas as pd
import csv


def bigquery_to_pandas(query, project='gabinete-compartilhado', credentials_file='/home/skems/gabinete/projetos/keys-configs/gabinete-compartilhado.json'):
    """
    Run a query in Google BigQuery and return its results as a Pandas DataFrame. 

    Input
    -----

    query : str
        The query to run in BigQuery, in standard SQL language.
    project : str
        
    
    Given a string 'query' with a query for Google BigQuery, returns a Pandas 
    dataframe with the results; The path to Google credentials and the name 
    of the Google project are hard-coded.
    """

    import google.auth
    import os

    # Set authorization to access GBQ and gDrive:
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_file

    
    credentials, project = google.auth.default(scopes=[
        'https://www.googleapis.com/auth/drive',
        'https://www.googleapis.com/auth/bigquery',
    ])
    
    return pd.read_gbq(query, project_id=project, dialect='standard', credentials=credentials)


def load_data_from_local_or_bigquery(query, filename, force_bigquery=False, save_data=True, 
                                     project='gabinete-compartilhado', 
                                     credentials_file='/home/skems/gabinete/projetos/keys-configs/gabinete-compartilhado.json',
                                     low_memory=False):
    """
    Loads data from local file if available or download it from BigQuery otherwise.
    
    
    Input
    -----
    
    query : str
        The query to run in BigQuery.
    
    filename : str
        The path to the file where to save the downloaded data and from where to load it.
        
    force_bigquery : bool (default False)
        Whether to download data from BigQuery even if the local file exists.
        
    save_data : bool (default True)
        Wheter to save downloaded data to local file or not.
        
    project : str (default 'gabinete-compartilhado')
        The GCP project where to run BigQuery.
        
    credentials_file : str (default path to 'gabinete-compartilhado.json')
        The path to the JSON file containing the credentials used to access GCP.
        
    low_memory : bool (default False)
        Whether or not to avoid reading all the data to define the data types
        when loading data from a local file.

    Returns
    -------
    
    df : Pandas DataFrame
        The data either loaded from `filename` or retrieved through `query`.
    """
    
    # Download data from BigQuery and save it to local file:
    if os.path.isfile(filename) == False or force_bigquery == True:
        print('Loading data from BigQuery...')
        df = bigquery_to_pandas(query, project, credentials_file)
        if save_data:
            print('Saving data to local file...')
            df.to_csv(filename, quoting=csv.QUOTE_ALL, index=False)
    
    # Load data from local file:
    else:
        print('Loading data from local file...')
        df = pd.read_csv(filename, low_memory=low_memory)
        
    return df


def upload_to_storage_gcp(bucket, key, data, project='gabinete-compartilhado'):
    """
    Given a data bucket (e.g. 'brutos-publicos') a key (e.g. 
    'executivo/federal/servidores/data/201901_Cadastro.csv'), 
    and 'data' (a string with all the data), write to GCP storage.
    """

    from google.cloud import storage
    
    storage_client = storage.Client(project=project)

    bucket = storage_client.get_bucket(bucket)
    blob = bucket.blob(key)

    blob.upload_from_string(data)


def upload_single_file(filename, bucket, key, verbose=False, project='gabinete-compartilhado', credentials='/home/skems/gabinete/projetos/keys-configs/gabinete-compartilhado.json'):
    """
    Upload file `filename` (str) to Google Storage's `bucket` 
    under a 'key' (str).
    """
    
    # Read file:
    if verbose:
        print('Reading file...')
    with open(filename, 'r') as f:
        text = f.read()
    
    # Upload file:
    if verbose:
        print('Uploading data to GCP...')
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials 
    upload_to_storage_gcp(bucket, key, text, project)


def upload_single_percent_partitioned(filename, local_folder, remote_folder, bucket='brutos-publicos', project='gabinete-compartilhado', 
                                      credentials='/home/skems/gabinete/projetos/keys-configs/gabinete-compartilhado.json', verbose=True):
    """
    Upload a file to GCP storage, at a directory
    tree described by the filaname's '%' simbols.
    E.g.:
    'data=2021-01-01%secao=2%artigo001.json' goes to
    'data=2021-01-01/secao=2/artigo001.json'.
    
    Parameters
    ----------
    filename : str
        Path to file to be uploaded, that might 
        include '%' separators used as folder 
        indicators for the file's location at 
        GCP.
    local_folder : str
        Part of the path `filename` to be replaced 
        with `remote_folder`. The root location of 
        `filename` locally.
    remote_folder : str
        Where in GCP storate to place `filename`,
        used as replacement for `local_folder`.
    bucket : str
        The GCP storage bucket.
    project : str
        The GCP project owner of the `bucket`.
    credentials : str
        Path to a JSON file containing GCP credentials
        for `project`.
    """
    # Upload to GCP:
    key = filename.replace('%', '/').replace(local_folder, remote_folder)
    upload_single_file(filename, bucket, key, verbose, project, credentials)


def upload_many_percent_partitioned(file_list, local_folder, remote_folder, bucket='brutos-publicos', project='gabinete-compartilhado', 
                                    credentials='/home/skems/gabinete/projetos/keys-configs/gabinete-compartilhado.json', verbose=True):
    """
    Upload all files supplied to GCP storage, at a directory
    tree described by each filaname's '%' simbols.
    E.g.:
    'data=2021-01-01%secao=2%artigo001.json' goes to
    'data=2021-01-01/secao=2/artigo001.json'.
    
    Parameters
    ----------
    file_list : list of str
        Paths to files to be uploaded, which might 
        include '%' separators used as folder 
        indicators for the file's location at 
        GCP.
    local_folder : str
        Part of each path in `file_list` to be replaced 
        with `remote_folder`. The root location of 
        the files locally.
    remote_folder : str
        Where in GCP storate to place each file,
        used as replacement for `local_folder`.
    bucket : str
        The GCP storage bucket.    
    project : str
        The GCP project owner of the `bucket`.
    credentials : str
        Path to a JSON file containing GCP credentials
        for `project`.
    """
    for filename in file_list:
        if verbose is True:
            print(filename)
        upload_single_percent_partitioned(filename, local_folder, remote_folder, bucket, project, credentials, verbose)
