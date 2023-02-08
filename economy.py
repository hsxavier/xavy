#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 12:18:01 2021

@author: skems
"""

import pandas as pd
import xavy.clouds as cl
import numpy as np
from sklearn.metrics import pairwise_distances


def load_ipca(ipca_file, verbose=True):
    """
    Load IPCA data from `ipca_file` (str) as a DataFrame.
    If `ipca_file` does not exist, load it from BigQuery.
    """
    # Load IPCA:
    ipca_df = cl.load_data_from_local_or_bigquery('SELECT * FROM `gabinete-compartilhado.bruto_gabinete_administrativo.ipca`', ipca_file)
    ipca_df['mes'] = pd.to_datetime(ipca_df['mes'])
    if verbose:
        print('Last IPCA date: ', ipca_df['mes'].max().date())

    return ipca_df


def project_ipca(ipca, final_date):
    """
    Return a DataFrame that extrapolates 
    `ipca` (DataFrame containing months and
    IPCA in decreasing chronological order)
    up to date `final_date`.
    """
    
    # Compute the average IPCA fractional increase in the last 12 months:
    avg_ipca = (ipca['ipca'] / ipca['ipca'].shift(-1)).iloc[:12].mean()
    
    # Build a monthly sampled dates, from last one in `ipca` up to `final_date`:
    last_date = ipca['mes'].max()
    dates = pd.date_range(last_date, final_date, freq='M') + pd.DateOffset(days=1)
    
    # Use avg. IPCA to extrapolate:
    last_ipca = ipca.loc[ipca['mes'] == last_date, 'ipca'].iloc[0]
    proj_ipca = last_ipca * pd.Series([avg_ipca] * len(dates)).cumprod()

    # Build DataFrame:
    proj_df = pd.DataFrame()
    proj_df['mes'] = dates
    proj_df['ipca'] = proj_ipca
    proj_df = proj_df.sort_values('mes', ascending=False)
    
    return proj_df


def complement_ipca(ipca, final_date):
    """
    Extend `ipca` (DataFrame) up to 
    `final_date`.
    """
    # Get extrapolation for IPCA:
    proj  = project_ipca(ipca, final_date)
    # Concatenate to existing data:
    compl = pd.concat([proj, ipca], ignore_index=True)
    
    return compl


def load_extended_ipca(ipca_file, final_ipca_date, verbose=True):
    """
    Load IPCA data from `ipca_file` (str) as a DataFrame.
    If `ipca_file` does not exist, load it from BigQuery.
    If the last available date is less than `final_ipca_date`
    (str %Y-%m-%d or date-like object), extrapolate IPCA
    up to `final_ipca_date` with the average increase rate
    over the last 12 months.
    """
    
    # Load IPCA:
    ipca_df = load_ipca(ipca_file, verbose)
    
    # Extrapolate IPCA:
    ipca_df = complement_ipca(ipca_df, final_ipca_date)

    return ipca_df


def deflate_values(date_series, value_series, ipca_df, ref_date=None, suffix='_deflac'):
    """
    Compute IPCA-corrected values.
    
    Parameters
    ----------
    date_series : Series
        A Pandas Series of dates (the day should always
        be the first).
    value_series : Series
        Values to be deflated by IPCA, aligned with the 
        dates in `date_series`.
    ipca_df : DataFrame
        DataFrame with columns 'mes' (datetime, where the
        day is always the first) and 'ipca' (the IPCA 
        index for the corresponding month).
    ref_date : str, datetime or None.
        Reference date (if str, in format %Y-%m-%d) for
        the deflated values. If None, use the last date
        in `ipca_df` as reference.
    suffix : str
        A suffix for the returned Series.
        
    Returns
    -------
    deflac_values : Series
        A series with deflated version of values in 
        `value_series` and name given by `value_series`
        name + `suffix`.
    """
    
    assert len(date_series) == len(value_series), '`date_series` and `value_series` should have the same length.'
    assert (date_series.index == value_series.index).all(), '`date_series` and `value_series` should have the same index.'
    
    # Find reference IPCA:
    if ref_date != None:
        ref_ipca = ipca_df.loc[(ipca_df['mes'] == ref_date), 'ipca']
        if len(ref_ipca) == 0:
            raise Exception('{} not found in `ipca_df`.'.format(ref_date))
        ref_ipca = ref_ipca.iloc[0]
    # If no ref. date was provided, use the last one:
    else:
        ref_ipca = ipca_df.sort_values('mes', ascending=False)['ipca'].iloc[0]
    assert ref_ipca != None, 'Reference IPCA is null.'
    
    # Join IPCA DataFrame to data:
    df = pd.DataFrame({'mes': date_series, value_series.name: value_series})
    assert len(df) == len(date_series), 'DataFrame of dates and values have more rows than input.'
    df = df.merge(ipca_df, how='left', on='mes')
    assert len(df) == len(date_series), 'Joining IPCA to input data led to a change in the number of rows'
    
    # Deflate values:
    series = df[value_series.name] / df['ipca'] * ref_ipca
    series.name = value_series.name + suffix
    
    return series


def gini_coefficient(X):
    """
    Compute the Gini coefficient for the 
    sample `X`.
    
    Parameters
    ----------
    X : 1D array-like
        The data to compute the Gini measurement
        of equality for.
    
    Returns
    -------
    gini : float
        The gini coefficient.
    """

    # Standardizing input:
    X = np.array(X)
    n = len(X)
    assert len(X.shape) == 1, 'Expecting 1D arrays.'
    
    return pairwise_distances(X.reshape((n, 1)), metric='l1').sum(axis=None) / (2 * len(X) * np.sum(X))
