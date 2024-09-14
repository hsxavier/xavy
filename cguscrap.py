#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Functions for scraping data from 
CGU's Portal da Transparência: http://www.portaltransparencia.gov.br
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

from selenium.webdriver import Firefox
from selenium import webdriver

from bs4 import BeautifulSoup
from time import sleep
from glob import glob
import re
import datetime as dt
from pathlib import Path

import xavy.dataframes as xd


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


#################################################################
### Captura de dados orçamentários do Portal da Transparência ###
#################################################################


def capture_detalhar_links(driver, link_class='linkRendered', root_url='http://www.portaltransparencia.gov.br'):
    """
    Given a Selenium driver that loaded a consulta at 
    Portal da Transparência, get the 'Detalhar' links 
    for all elements in the table.
    
    Returns a list of links
    """
    source = driver.page_source
    soup = BeautifulSoup(source, features='lxml')
    detalhar_links = find_all_class(soup, link_class)
    links = [root_url + a.attrs['href'] for a in detalhar_links]
    
    return links


def check_pagination(driver, next_class='next', stop_class='disabled'):
    """
    Return the pagination 'next' button if there is another
    data page to load at a consulta in Portal da Transparência.
    If there are no pages to load next, return None.    
    """
    
    next_button = driver.find_element_by_class_name(next_class)
    next_button_class = next_button.get_attribute('class')
    pagination_end = next_button_class.find(stop_class) != -1
    
    if pagination_end == True:
        return None
    else:
        return next_button


def get_all_detalhar_links(url, wait_load_sec=10, wait_page=5, verbose=True):
    """
    Given an `url` (str) to a 'Consulta' at Portal da Transparência 
    on 'Despesas', return all the links to 'Detalhar' present 
    in the paginated table of despesas.
    """
    # Load driver and webpage:
    if verbose:
        print('Starting driver...')
    driver = Firefox()
    if verbose:
        print('Loading URL...')
    driver.get(url)
    sleep(wait_load_sec)
    
    # Start capturing 'Detalhar' links:
    if verbose:
        print('Scraping links...')
    page = 1
    links = capture_detalhar_links(driver)
    next_button = check_pagination(driver)
    if verbose:
        print('Pág. {}'.format(page), end='  ')
    
    # Continue capture by paginating:
    while next_button is not None:
        next_button.click()
        sleep(wait_page)
        links = links + capture_detalhar_links(driver)
        next_button = check_pagination(driver)
        page += 1
        if verbose:
            print('Pág. {}'.format(page), end='  ')
        
    
    # Close driver:
    if verbose:
        print('')
        print('Closing driver...')
    driver.close()
    driver.quit()

    return links


def start_download_driver(download_dir='/home/skems/Downloads/', filetype='text/plain'):
    """
    Start a selenium driver that automatically downloads files of 
    type `filetype` (str) to folder `download_dir` (str).
    
    Returns a driver (browser).
    """
    
    # To prevent download dialog:
    profile = webdriver.FirefoxProfile()
    profile.set_preference('browser.download.folderList', 2) # custom location
    profile.set_preference('browser.download.manager.showWhenStarting', False)
    profile.set_preference('browser.download.dir', download_dir)
    profile.set_preference('browser.helperApps.neverAsk.saveToDisk', filetype)
    
    # Start driver:
    driver = webdriver.Firefox(profile)
    
    return driver


def fmt_date(date):
    """
    Format a datetime `date` into str in 
    Portal da Transparência format.
    """
    return date.strftime('%d/%m/%Y')


def split_url_by_date(url):
    """
    Take a URL (str) for the details of a Despesa in Portal da Transparência
    and split it into two equal-length date ranges.
    
    Returns a list of two urls (str).
    """
    # Get URL dates:
    match_obj  = re.search('de=(\d{2}/\d{2}/\d{4})', url)
    start_date = dt.datetime.strptime(match_obj.group(1), '%d/%m/%Y')
    match_obj  = re.search('ate=(\d{2}/\d{2}/\d{4})', url)
    end_date   = dt.datetime.strptime(match_obj.group(1), '%d/%m/%Y')
    # Create new dates:
    middle_date = start_date + (end_date - start_date) / 2
    next_date   = middle_date + dt.timedelta(days=1)

    assert middle_date != end_date
    assert next_date != start_date
    
    # Split URL into new dates:
    new_url_1 = url.replace('ate={}'.format(fmt_date(end_date)), 'ate={}'.format(fmt_date(middle_date)))
    new_url_2 = url.replace('de={}'.format(fmt_date(start_date)), 'de={}'.format(fmt_date(next_date)))
    
    return [new_url_1, new_url_2]


def get_retry_urls(failed_download):
    """
    Get a list of dicts containing URLs under the key
    url and return two new URLs for each original one
    by splitting the date into two disjoint and complete
    date ranges.
    
    Returns a list of URLs (str).
    """
    
    retry_urls = []
    for d in failed_download:
        retry_urls = retry_urls + split_url_by_date(d['url'])
        
    return retry_urls


def download_click_files_from_urls(urls, download_dir='/home/skems/Downloads/', wait_load_sec=10, wait_download_sec=5, verbose=True):
    """
    Download data from Consultas of Portal da Transparência by
    clicking on the "Baixar" button.
    
    Parameters
    ---------- 
    urls : list of str
        URLs to the Portal da Transparência's consultas.   
    download_dir : str
        Full path to the folder where to save the downloaded files.
        It must be empty.   
    wait_load_sec : float
        Number of seconds to wait after getting an URL with 
        the driver.    
    wait_download_sec : float
        Number of seconds to wait after clicking the 'Baixar'
        button.   
    verbose : bool
        Print messages or not.
        
    Returns
    -------
    failed_download : list of dicts
        Data about the downloads that failed.
    """
    
    n_files = len(glob(download_dir + '*'))
    assert n_files == 0, '`download_dir` should be empty.'

    driver = start_download_driver(download_dir)

    # Loop over links:
    page = 0
    failed_download = []
    for url in urls:
        page += 1
        if verbose:
            print('Pág. {}'.format(page), end='  ')

        # Loads page:
        driver.get(url)
        sleep(wait_load_sec)

        # Click to download:
        baixar_button = driver.find_element_by_id('btnBaixar')
        baixar_button.click()
        sleep(wait_download_sec)
        n_files = len(glob(download_dir + '*'))
        if n_files + len(failed_download) != page:
            failed_download.append({'i': page - 1, 'page': page, 'n_files': n_files, 'url': url})
            print('(fail)', end='  ')
        
    # Close the driver:
    driver.close()
    driver.quit()
    
    return failed_download


def concat_ptransp_scraps(docs_dir):
    """
    Join scraped data from Portal da Transparência, stored in 
    'documentos*.csv' files located at `docs_dir` (str) into 
    a single DataFrame, which is returned.
    """
    files = Path(docs_dir).rglob('documentos*.csv')
    scrap = pd.concat([pd.read_csv(f, sep=';', encoding='utf-8') for f in files], ignore_index=True)
    return scrap


def download_detalhes(root_download_dir, urls, verbose=True):
    """
    Baixa dados detalhados de execução de despesas do governo
    federal, a partir de links do Portal da Transparência.
    
    Parâmetros
    ----------
    root_download_dir : str 
        Caminho completo para o diretório onde salvar os 
        dados de detalhamento baixados. Este diretório 
        precisa estar vazio. Caso os links em `urls` 
        resultem em mais de 1000 detalhamentos, estes serão 
        salvos em sub-diretórios separados de `root_download_dir`. 
    urls : array-like of str
        URLs para os detalhamentos da execução orçamentária.
    """
    # Primeiro download:
    print(xd.bold('Primeira carga'))
    failed_download = download_click_files_from_urls(urls, root_download_dir)

    # Try to get the failed downloads:
    retry_counts = 0
    while len(failed_download) > 0:

        # Get a new dir:
        retry_counts += 1
        download_dir = root_download_dir + 'r{:02d}/'.format(retry_counts)
        # Split the URLs:
        retry_urls = get_retry_urls(failed_download)
        print(xd.bold('\nRetry #{} : {} urls -> {}'.format(retry_counts, len(retry_urls), download_dir)))

        # Retry:
        failed_download = download_click_files_from_urls(retry_urls, download_dir)
