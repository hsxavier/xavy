import pandas as pd
import numpy as np
import geopandas as gp
import matplotlib.pyplot as pl
import pycep_correios as cep
import geopy.geocoders as geo
import requests
from scipy.spatial import cKDTree
from shapely.geometry import Point
from time import sleep
from urllib.parse import quote


#################
### Geocoding ### 
#################

def cep_to_addr_dict(cep_str, webservice=cep.WebService.CORREIOS, verbose=True, raise_error=True, sleep_time=1, max_retries=4):
    """
    Given a CEP `cep_str` (str) in '00000-000' format,
    return a dict with address data. If this translation
    fails, returns an dict with null values.
    
    webservice can be APICEP, VIACEP ou CORREIOS.
    """

    # Hard-coded:
    null_addr = {'bairro': np.NaN,
             'cep': np.NaN,
             'cidade': np.NaN,
             'logradouro': np.NaN,
             'uf': np.NaN,
             'complemento': np.NaN}
 
    # Start:
    n_try = 0
    if verbose:
        print('  {}'.format(cep_str), end='')
    addr = null_addr

    # Exit if input is null:
    if type(cep_str) == float or cep_str is None:
        return null_addr
    
    # Loop over attempts:
    while n_try < max_retries and addr == null_addr:
        n_try += 1
        if verbose:
            print('.', end='')
            
        # Try to get the address and ignore errors:
        if raise_error == False or n_try < max_retries:
            try:
                addr = cep.get_address_from_cep(cep_str, webservice=webservice)
            except:
                addr = null_addr

        # Raise error if attempt fails:
        else:
            addr = cep.get_address_from_cep(cep_str, webservice=webservice)               
    
        sleep(sleep_time)
    
    if verbose == True and addr == null_addr:
        print(' FAIL', end='')
    
    return addr


def address_to_coord_dict(address, geocoder='nominatim', key_file='/home/skems/gabinete/projetos/keys-configs/google_geocode_api_key.txt', user_agent="Gambiarra25", verbose=True):
    """
    Geocode (translate to coordinates) an address using a web API.
    
    Parameters
    ----------
    address : str
        Address (e.g. R. Bela Vista, 398, Aclimação - São Paulo, SP)
        to get the latitude and longitude to.
    geocoder : str
        Name of the web API used to geocode. Options are:
        -- 'nominatim', for the open service that uses OpenStreetMap;
        -- 'google', for the paid service Google Geocoding API
           (requires an API key).
    key_file : str
        Path to the file containing the Google API key. Only used
        if `geocoder='google'`.
    user_agent : str
        Username for the 'nominatim' option.
        
    Returns
    -------
    coords : dict from str to float
        A dict with keys 'latitude' and 'longitude', pointing
        to their respective coordinates.
    """
    
    if type(address) == float or address is None:
        return {'latitude': np.NaN, 'longitude': np.NaN}
    
    if geocoder == 'nominatim':
        geolocator = geo.Nominatim(user_agent=user_agent)
    elif geocoder == 'google':
        with open(key_file, 'r') as f:
            api_key = f.read().replace('\n', '')
        geolocator = geo.GoogleV3(api_key=api_key)
    
    location = geolocator.geocode(address)
    if location is None:
        location = geolocator.geocode(address.split(' - ')[0])
    if location is None:
        return {'latitude': np.NaN, 'longitude': np.NaN}
        
    return {'latitude': location.latitude, 'longitude': location.longitude}


def query_google_places(query, countries='country:br', lang='pt_BR', key_file='/home/skems/gabinete/projetos/keys-configs/google_geocode_api_key.txt'):
    """
    Send a query to Google Places API and get the 
    results.
    
    Parameters
    ----------
    query : str
        The query to be made for a place.
    countries : str
        Countries where to look for the place. 
        It sohuld have the form 'country:XX' 
        where XX is an ISO_3166-1 country code 
        like 'br'. Up to 5 countries may be 
        selected with pipe, e.g.: 
        'country:us|country:fr'
    lang : str
        Language in which to return the 
        results.
    key_file : str
        Path to the file containing the 
        Google Places API key.
    
    Returns
    -------
    response : HTTP requests response
        Object containing the API's 
        response at `response.text`.
        
    More info at:
    https://developers.google.com/maps/documentation/places/web-service/autocomplete
    """
    # Hard-coded:
    url_template = "https://maps.googleapis.com/maps/api/place/autocomplete/json?input=%(input)s&components=%(components)s&language=%(lang)s&key=%(api_key)s"
    payload = {}
    headers = {}

    # Get API key:
    with open(key_file, 'r') as f:
        api_key = f.read().replace('\n', '')
    
    # Build request URL:
    url_params = {'input': quote(query), 'components': quote(countries), 'api_key': api_key, 'lang': lang}
    url = url_template % url_params
    
    # Make request:
    response = requests.get(url, headers=headers, data=payload)
    
    return response


###########################
### GeoPandas functions ###
###########################


def extract_coordinates(geoseries):
    """
    Returns a DataFrame with 2D coordinates of 
    points specified in `geoseries`.
    """
    coord_df = pd.DataFrame()
    coord_df['x'] = geoseries.x
    coord_df['y'] = geoseries.y
    return coord_df


def find_nearest(gdA, gdB):
    """
    For each point location in `gdA` (GeoDataFrame), 
    find the location in `gdB` (GeoDataFrame) closest
    to it.
    
    Both GeoDataFrames are expected to have one 
    'geometry' column containing POINTS.
    
    Returns `gdA` with extra columns from `gdB`, 
    except its 'geometry', plus a distance column.
    """
    
    nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    gdB_nearest = gdB.iloc[idx].drop(columns="geometry").reset_index(drop=True)
    gdf = pd.concat([gdA.reset_index(drop=True), gdB_nearest, pd.Series(dist, name='dist')], axis=1)

    return gdf


def load_wkt_csv(filename, geometry_col='geometry', missing_geo='remove', crs='epsg:4326', drop_src=True, **kwargs):
    """
    Load a CSV file containing WKT (Well-known text 
    representation of geometry) strings among its 
    columns as a GeoDataFrame.
    
    Parameters
    ----------
    filename : str
        The path to the CSV file to load.
    geometry_col : str
        The name of the column containing the 
        geometry.
    missing_geo : str
        A WKT string used to fill missing geometry
        values or 'remove' in order to drop these 
        rows.
    crs : str
        The coordinate reference system used in 
        the input geometry WKT (defaults to latitude
        and longitude in degrees from Greenwich 
        meridian).
    drop_src : bool
        Whether to keep or remove the original 
        `geometry_col`. If the latter is 'geometry',
        keep the column (but converted to geometry).
    kwargs : keyword arguments
        Arguments for the Pandas df.read_csv() method.
        
    Returns
    -------
    gdf : GeoDataFrame
        GeoDataFrame with all columns from `filename`,
        with `geometry_col` converted to geometry and 
        named 'geometry'. If `drop_src` is False, the 
        original `geometry_col` (as str or object) is 
        kept along with the 'geometry' one.
        
    """
    
    from shapely import wkt
    
    # Load CSV as a Pandas DataFrame:
    pre_df = pd.read_csv(filename, **kwargs)
    
    # Remove missing geometries if requested:
    if missing_geo == 'remove':
        pre_df = pre_df.loc[~pre_df[geometry_col].isnull()]
    
    # Create geometry column:
    pre_df['geometry'] = pre_df[geometry_col].fillna(missing_geo).astype(str).apply(wkt.loads)
    # Drop original column:
    if geometry_col != 'geometry' and drop_src is True:
        pre_df.drop(columns=geometry_col, inplace=True)
    
    # Convert to GeoPandas:
    gdf = gp.GeoDataFrame(pre_df, crs=crs)
    
    return gdf


def add_label_to_map(gdf, labels, disperse=0, ha='center', va='center', ax=None, **kwargs):
    """
    Show the labels of regions on the map.
    
    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing the geometry for 
        which to show the labels.
    labels : array-like or str
        The labels of each geometry in `gdf`, 
        in the same order, or the name of 
        the column in `gdf` containing the 
        labels.
    disperse : float
        If > 0, the standard deviation of 
        random displacements of the labels 
        around the representative points of 
        the geometries, in the same units as 
        the geometries' coordinates.
    """
    # Standardize input to iterable:
    if type(labels) == str:
        labels = gdf[labels]

    # Get region's representative points:
    loc = gdf.representative_point()
    
    # Generate displacements for labels:
    if disperse > 0:
        ex = np.random.normal(0, disperse, len(loc))
        ey = np.random.normal(0, disperse, len(loc)) 
    else:
        ex = np.zeros_like(loc)
        ey = np.zeros_like(loc)
    
    # Add labels to map:
    for t, x, y in zip(labels, loc.x + ey, loc.y + ey):
        if ax is None:
            pl.annotate(t, (x, y), ha=ha, va=va, **kwargs)
        else:
            ax.annotate(t, (x, y), ha=ha, va=va, **kwargs)
