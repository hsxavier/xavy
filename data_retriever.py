# Functions for downloading data (and unzipping it).

import requests
import zipfile
import io


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


def retrieve_zipped_files(url, save_dir, verbose=True, timeout=10800, keep_zip_dir=True):
    """
    Downloads a ZIP file and unzip it.
    
    Input
    -----
    
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
    
    assert type(timeout) == int and timeout > 0, '`timeout` should be a int > 0.'
    assert type(url) == str, '`url` should be a str.'
    assert type(save_dir) == str, '`save_dir` should be a str.'
    assert url[-4:].lower() == '.zip', 'Expecting ZIP file.'

    if verbose:
        print('Downloading file...')
    session = requests.session()
    session.mount('http://', requests.adapters.HTTPAdapter(max_retries=3))
    response = session.get(url, timeout=timeout)

    if response.status_code != 200:
        raise Exception('HTTP request failed with code ' + str(response.status_code))
    
    if verbose:
        print('Unzipping file...')
    z = zipfile.ZipFile(io.BytesIO(response.content))

    if keep_zip_dir == True:
        save_dir = include_zip_dir(save_dir, url)
    z.extractall(save_dir)
    if verbose:
        print('Files unzipped to ' + save_dir)
    