#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Functions for downloading data (and unzipping it).
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

import requests
import zipfile
import io
import os
import gzip
import shutil
from glob import glob


def set_slash(directory):
    
    assert type(directory) == str, '`directory` should be a str.'
    
    if directory[-1] != '/':
        return directory + '/'
    else:
        return directory

    
def concat_path(path_parts_list):
    """
    Joins a list of parts of path (sub-directories and file) into a full path.
    """
    
    assert type(path_parts_list) == list and type(path_parts_list[0] == str), \
    '`path_parts_list` should be a list of str.'
    
    full_path = ''.join([set_slash(path_part) for path_part in path_parts_list])
    return full_path[:-1]


def include_zip_dir(root_dir, url):
    """
    Create a new directory path (str) by adding the name of the zip file 
    stored in `url` (str) to `root_dir`, e.g.:
        `../dados/` + `http://test.org/file.zip` = `../dados/file` 
    """
    return concat_path([root_dir, url.split('/')[-1][:-4]])


def gzip_decompress(filename):
    """
    Decompress gzip file in path `filename` (str) as the shell command
    `gzip -d filename` would.
    """

    # Security check:
    extension = filename.split('.')[-1]
    assert extension == 'gz', 'Expecting ".gz" extension, found .{:}.'.format(extension)

    # Create decompressed name:
    out_name = '.'.join(filename.split('.')[:-1])
    # Unzip and read file:
    with gzip.open(filename, 'rb') as f_in:
        # Create output file:
        with open(out_name, 'wb') as f_out:
            # Copy to output file:
            shutil.copyfileobj(f_in, f_out)

    # Delete compressed file:
    os.remove(filename)

def retrieve_zipped_files(url, save_dir, verbose=True, timeout=10800, keep_zip_dir=True):
    """
    Downloads a ZIP file and unzip it.
    
    Parameters
    ----------
    url : str
        The URL address of the file to download.        
    save_dir : str
        The path to the folder where to save the unzipped files. New 
        folders are created as needed.      
    verbose : bool (default True)
        Whether or not to print status messages along the process.    
    timeout : int (detault 10800)
        Number of seconds to wait for download before giving up. Default 
        is 3 hours.    
    keep_zip_dir : bool (default True)
        Wheter or not to unzip the content of the zip file into a folder of same name
        (inside `save_dir` folder).
        
    Returns
    -------
    Nothing
    """

    # Security checks:
    assert type(timeout) == int and timeout > 0, '`timeout` should be a int > 0.'
    assert type(url) == str, '`url` should be a str.'
    assert type(save_dir) == str, '`save_dir` should be a str.'
    extension = url.split('.')[-1]
    assert extension in {'gz', 'zip'}, 'Expecting ".gz" or ".zip" extension, found .{:}.'.format(extension)
    #assert extension in {'zip'}, 'Expecting ".zip" extension, found .{:}.'.format(extension)
    # GZIP handling not implemented.

    # Download:
    if verbose:
        print('Downloading file...')
    session = requests.session()
    session.mount('http://', requests.adapters.HTTPAdapter(max_retries=3))
    response = session.get(url, timeout=timeout)
    # If fail:
    if response.status_code != 200:
        raise Exception('HTTP request failed with code ' + str(response.status_code))

    # Decompressing:
    # If ZIP:
    if extension == 'zip':
        if verbose:
            print('Unzipping file...')
        z = zipfile.ZipFile(io.BytesIO(response.content))
        if keep_zip_dir == True:
            save_dir = include_zip_dir(save_dir, url)
        z.extractall(save_dir)
        if verbose:
            print('Files unzipped to ' + save_dir)

    # If GZIP:
    elif extension == 'gz':
        zipname = url.split('/')[-1]
        zippath = concat_path([save_dir, zipname])
        if verbose:
            print('Saving gzip file...')
        with open(zippath, 'wb') as fzip:
            fzip.write(response.content)
        if verbose:
            print('Decompressing file...')
        gzip_decompress(zippath)
        if verbose:
            print('File decompressed to ' + save_dir)



def filestem(filename):
    """
    Returns the stem (str) of the `filename` (str), i.e. everything 
    up to the last dot that usually marks the file extension
    """
    stem = '.'.join(filename.split('.')[:-1])
    return stem


def sync_remote_zipped_files(url, save_dir, verbose=True, timeout=10800, keep_zip_dir=True, force_download=False):
    """
    Downloads a ZIP file and unzip it if requested or if not locally present.
    
    Parameters
    ----------
    url : str
        The URL address of the file to download.        
    save_dir : str
        The path to the folder where to save the unzipped files. New 
        folders are created as needed.      
    verbose : bool (default True)
        Whether or not to print status messages along the process.    
    timeout : int (default 10800)
        Number of seconds to wait for download before giving up. Default 
        is 3 hours.    
    keep_zip_dir : bool (default True)
        Whether or not to unzip the content of the zip file into a folder of same name
        (inside `save_dir` folder).
    force_download : bool
        If True, download the file again and overwrite it. If False, do not
        download the file if it is already present.
        
    Returns
    -------
    Nothing
    """
    
    # Get local file pattern:
    zip_filename = url.split('/')[-1]
    stem = filestem(zip_filename)
    file_pattern = os.path.join(save_dir, stem)
    # Get matching files:
    matching_files = glob(file_pattern) + glob(file_pattern + '.*')

    # Warning (more than one match, I don't know what to do):
    if len(matching_files) > 1 and verbose is True:
        # More than one match, I don't know what to do:
        print('!! Found more than one matching file or folder.')
        print(matching_files)
        
    # Found local file:
    if len(matching_files) > 0:
        # Download file:
        if force_download is True:
            print('Found a local file, will overwrite.')
            retrieve_zipped_files(url, save_dir, verbose=verbose, timeout=timeout, keep_zip_dir=keep_zip_dir)
        else:
            print('Found a local file, skip download.')
    else:
        print('No local file found.')
        retrieve_zipped_files(url, save_dir, verbose=verbose, timeout=timeout, keep_zip_dir=keep_zip_dir)
