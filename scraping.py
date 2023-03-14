#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Functions for web scraping
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

import time
from datetime import date
import os
from glob import glob
from selenium.webdriver import Firefox
from bs4 import BeautifulSoup


###############################
### Retrieving source pages ###
###############################

def get_page_source(url):
    """
    Use Selenium to download the source of the webpage 
    at `url` (str). Return it as a str.
    """
    
    # Get page source:
    driver = Firefox()
    driver.get(url)
    source = driver.page_source
    
    driver.close()
    driver.quit()

    return source


def save_raw_scrap_data(content, file, create_dirs=True):
    """
    Write string `content` to file with name `file` (str).
    If create_dirs is True, create any required directories
    in the path `file`.
    """
    
    if create_dirs == True:
        dirs = os.path.dirname(file)
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        
    with open(file, 'w') as f:
        f.write(content)

    
def gen_filename(file_prefix, page, zfill=2, add_date=True):
    """
    Generate filename by combining `file_prefix` (str, with path),
    `page` (int) and date, if requested.
    
    Parameters
    ----------
    file_prefix : str
        Prefix of the file, containing the path.
    page : int
        An integer representing the page.
    zfill : int
        Total number of digits in page, to be filled 
        with zeros.
    add_date : bool
        Whether to add the current date to the filename.
        
    Returns
    -------
    filename : str
        The filename built.
    """
    # Date:
    if add_date == True:
        today = date.today().strftime('%Y-%m-%d')
    else:
        today = ''
    
    # Page:
    p = str(page).zfill(zfill)
    
    # Filename:
    filename = '{}{}_p{}.html'.format(file_prefix, today, p)
    
    return filename
    
 
    
def scrap_multiple_url_pages(url_template, first_page, last_page, file_prefix='./',
                             zfill=2, sleep=0, save=True, gen_list=True, add_date=True):
    """
    Capture multiple webpages using Selenium where each page is accessed 
    with the same URL except for a different integer `page`.
    
    Parameters
    ----------
    url_template : str
        URL of the page to be captured, with a '{}' in place 
        of the page number (e.g. '.../casas/?pagina={}').
    first_page : int
        The first page to be scraped.
    last_page : int
        The last page to be scraped.
    file_prefix : str
        The path and file prefix where to save the source files 
        if `save` is True. To the prefix it is added the date 
        (if `add_date` is True) and the page.
    zfill : int
        Number of digits to represent the page, filled with zeros.
    sleep : float
        Number of seconds to sleep between each request.
    save : bool
        Whether to save the source codes to files.
    gen_list : bool
        Whether to return the sources in a list of str.
    add_date : bool
        Whether to add the date to the filename where the source is
        saved.
        
    Returns
    -------
    
    src_list : list of str or None
        List of the scraped sources (one entry per page), if `gen_list`
        is True. Otherwise, return None.
    """
    scrap_list = []
    for page in range(first_page, last_page + 1):
        
        # Get source:
        url = url_template.format(page)
        source = get_page_source(url)
        
        # Save to disk:
        if save == True:
            filename = gen_filename(file_prefix, page, zfill, add_date)
            save_raw_scrap_data(source, filename)
        
        # Append to list:
        if gen_list == True:
            scrap_list.append(source)
        
        time.sleep(sleep)
    
    if gen_list == True:
        return scrap_list
 
    
def load_file(filename):
    """
    Load the content of the file with `filename` (str)
    as a string.
    """
    with open(filename, 'r') as f:
        content = f.read()
    return content


def load_multiple_sources(file_pattern):
    """
    Load the contents of files with names following the glob 
    pattern `file_pattern` (str) as a list of str.
    """
    files = glob(file_pattern)
    contents = [load_file(f) for f in files]
    
    return contents


#########################
### Scraping webpages ###
#########################


def find_class(soup, class_attr):
    """
    Given a Beautiful Soup object `soup`, return its first element
    whose 'class' attribute is `class_attr` (str) as a Beautiful soup
    object.
    """
    
    return soup.find(attrs={'class':class_attr})


def get_class_text(soup, class_attr, strip=True):
    """
    Return the text inside the first element  in `soup` 
    (Beautiful Soup object) whose 'class' attribute 
    is `class_attr`. If `strip` is True, remove trailing 
    empty spaces.
    """
    element = find_class(soup, class_attr)
    if type(element) == type(None):
        return None
    
    text = element.text
    if strip == True:
        text = text.strip()
    
    return text


def find_all_class(soup, class_attr):
    """
    Given a Beautiful Soup object `soup`, return all its elements
    whose 'class' attribute is `class_attr` (str) as a list of 
    Beautiful soup objects.    
    """
    elements = soup.find_all(attrs={'class': class_attr})

    return elements


def source_to_class_elements(source, class_attr):
    """
    Given a HTML page source code `source` (str), return all 
    its elements whose 'class' attribute is `class_attr` (str) 
    as a list of Beautiful soup objects.    
    """
    
    soup = BeautifulSoup(source, features='lxml')
    elements = find_all_class(soup, class_attr)
    
    return elements
    
#url_template = 'https://www.zapimoveis.com.br/venda/casas/sp+sao-paulo+zona-norte+jd-s-paulo/?pagina={}'
#src_list = scrap_multiple_url_pages(url_template, 4, 24, '../dados/sources/zap-imoveis_venda_jd-sao-paulo/scrap_', sleep=15)