#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Functions for causal inference and causal diagrams.
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

import pygraphviz as gv
from PIL import Image
import io
from IPython.display import display
import pandas as pd
import numpy as np
from collections import defaultdict


#####################
### Plotting DAGs ###
#####################

def format_node(G, node_name, node_type, node_format={'target':'filled', 'hidden':'dashed'}):
    """
    Format in place a node in a graph according to the variable 
    type: hidden variable (unobserved) or target.
    
    Parameters
    ----------
    G : AGraph
        Graph created with pygraphviz containing the node
        to be formatted.
    node_name : str
        Name of the node to be formatted. This is the string
        drawn inside the ellipse.
    node_type : str
        Either 'target' or 'hidden'.
    node_format : dict
        From the node type (str) to the style (str) to be used
        for that node.
    """
    G.get_node(node_name).attr['style'] = node_format[node_type]

    
def plot_diagram(causal_links, special_nodes=None, rankdir='LR'):
    """
    Given a list of directed edges (represented
    by tuples of two elements), draws a graph.

    Parameters
    ----------
    causal_links : list of tuples
        List of directed edges (v1, v2) from node v1 to node v2.
    special_nodes : dict
        Two keys are allowed in the dict: 'hidden' and 'target'.
        The value of each key is a list of str containing names
        of nodes to be marked as such. Hidden have dashed contours,
        target have gray filling.
    """
    
    # Build graph:
    G = gv.AGraph(directed=True, rankdir=rankdir)
    G.add_edges_from(causal_links)
    
    # Formatting nodes, is requested:
    if special_nodes is not None:
        
        # Security checks:
        node_types = {'hidden', 'target'}
        assert type(special_nodes) is dict, '`special_nodes` should be dict, but are {}.'.format(type(special_nodes))
        unknown_keys = set(special_nodes.keys()) - node_types
        assert unknown_keys == set(), 'Found unknown node types {}.'.format(unknown_keys)
        
        # Loop over special node types:
        for node_type in special_nodes.keys():
        
            # Standardize `node_names` to list:
            node_names = special_nodes[node_type]
            if type(node_names) in (str, int):
                node_names = [node_names]
                
            # Format nodes of the given type:
            for node_name in node_names:
                format_node(G, node_name, node_type)
    
    # Draw graph:
    G.layout(prog='dot')
    img_bytes = G.draw(format='png')
    image = Image.open(io.BytesIO(img_bytes))
    display(image)


##############################
### Generating causal data ###
##############################

def generate_source(mean=0.0, dev=1.0, shift=0, type='normal', cat_probs=[0.5, 0.5], n_samples=10000):
    """
    Generate a sample of random variables.

    Parameters
    ----------
    mean : float
        Mean for a gaussian variable or the mean of the associated
        gaussian variable for a lognormal variable.
    dev : float
        Standard deviation of a Gaussian variable or the deviations
        of the associated Gaussian variable for a lognormal variable.
    shift : float
        The shift of a lognormal variable. The minimum value is 
        `-shift`.
    type : str
        Type of variable to generate. Either 'normal', 'lognormal'
        or 'categorical'.
    cat_probs : list-like of floats
        For categorical variables, this is the probability for 
        each category. It must add to one.
    n_samples : int
        How many instances of the variable to generate.
    
    Returns
    -------
    instances : array
        Instances of the specified random variable.
    """
    
    assert type in {'normal', 'lognormal', 'categorical'}, "`type` was '{:}', but must be 'normal', 'lognormal', or 'categorical'.".format(type)

    if type == 'categorical':
        total_prob = np.sum(cat_probs)
        assert np.isclose(total_prob, 1.0), '`cat_probs` is summing to {:} instead of 1.'
        choices = np.arange(len(cat_probs), dtype=int)
        return np.random.choice(choices, n_samples, p=cat_probs)
        
    if type == 'lognormal':
        return np.exp(np.random.normal(mean, dev, n_samples)) - shift
    
    if type == 'normal':
        return np.random.normal(mean, dev, n_samples)


def generate_dependent(listens_to, transform=None, noise=0.4):
    """
    Generate a sample of a dependent variable 
    from input variables `listens_to` (list of 
    arrays) by applying `transform` (function) to
    them and adding Gaussian noise with standard 
    deviation `noise` (float).
    """
    
    # Default transformation:
    if transform is None:
        transform = sumvar
    
    # Generate dependent variable and noise:
    y = transform(*listens_to)
    e = generate_source(0, noise, len(y))
    
    return y + e


def sumvar(*list_of_variable_samples):
    """
    Return a sample (array of floats) composed 
    of the element-wise  sum of the samples of 
    variables in `list_of_variable_samples` 
    (list of arrays of floats).
    """
        
    return np.sum(list_of_variable_samples, axis=0)


def compute_dependency(list_of_variables, y):
    """
    Compute the coefficients and their uncertainties 
    for the variables in `list_of_variables` (list of 
    arrays of floats) in a linear model for predicting
    `y`.
    
    Returns
    -------
    mean_coefs : array
        The coefficients for the variables in 
        `list_of_variables`, in the same order.
    dev_coeffts : array
        Uncertainties for the coefficients above.
    """
    # Prepare for modeling y as a function of input variables:
    X     = np.transpose(list_of_variables)
    model = LinearRegression()

    # LOOP over fit trials:
    coefs = []
    kf = KFold(n_splits=10)
    for train_idx, test_idx in kf.split(X):
        # Select subsample:
        train_X = X[train_idx]
        train_y = y[train_idx]
        # Fit the model:
        model.fit(train_X, train_y)
        # Store coefficients:
        coefs.append(model.coef_)

    # Compute coefficients and their errors:
    mean_coefs = np.mean(coefs,axis=0)
    dev_coefs  = np.std(coefs,axis=0)
    
    return mean_coefs, dev_coefs


def print_dependency(coefs, coefs_err):
    """
    Print the coefficients `coefs` (array)
    and their uncertainties `coefs_err` 
    (array) in the order they appear.
    """
    i = 0
    for m, e in zip(coefs, coefs_err):
        i += 1
        print('{}: {:6.3f} +/- {:6.3f}'.format(i, m, e))

        
def report_dependency(list_of_variables, y):
    """
    Compute and print the coefficients and 
    their uncertainties for the variables in 
    `list_of_variables` (list of arrays of floats) 
    in a linear model for predicting `y`.
    """
    print_dependency(*compute_dependency(list_of_variables, y))


def links2funcdict(causal_links):
    """
    Identify variables' dependencies in a list of causal links.
    
    Parameters
    ----------
    causal_links : list of tuples of str
        Each tuple (x, y) represents a causal relation x -> y.
        x and y (str) are names of variables. 

    Returns
    -------
    fdict : dict of set of str
        The keys are the names of dependent variables and the dict 
        values are the set of names of variables on which the 
        key depends on.
    """
    
    fdict = defaultdict(lambda: set())
    for link in causal_links:
        # Get sources `x` and dependent variables `y`:
        x = link[0]
        y = link[1]
        # Add `x` to `y` variables list:
        fdict[y].add(x)
        
    return dict(fdict)


def links2varset(causal_links, only_dep=False):
    """
    Get the set of variables present in the causal links.
    
    Parameters
    ----------
    causal_links : list of tuples of str
        Each tuple (x, y) represents a causal relation x -> y.
        x and y (str) are names of variables.
    only_dep : bool
        If True, only return dependent variables. If False,
        return all variables.

    Returns
    -------
    varset : set of str
        The set of names of variables (str) present in 
        `causal_links`. If `only_dep` is True, returns 
        only dependent variables; otherwise, return all.
    """
    
    varset = set()
    for x, y in causal_links:
        if only_dep is False:
            varset.add(x)
        varset.add(y)
    
    return varset


def links2sources(causal_links):
    """
    Find the set of independent variables (sources) in
    a list of causal links.
    
    Parameters
    ----------
    causal_links : list of tuples of str
        Each tuple (x, y) represents a causal relation x -> y.
        x and y (str) are names of variables.

    Returns
    -------
    src_set : set of str
        The set of names of independent variables (str) present in 
        `causal_links`.
    """
    all_set = links2varset(causal_links)
    dep_set = links2varset(causal_links, True)
    src_set = all_set - dep_set
    
    return src_set


def ready_to_create(source_list, created_vars):
    """
    Inform whether all variables in `source_list` (set) already 
    have been created (i.e. are in set `created_vars`) or not.
    """
    #return source_list - created_vars
    return len(source_list - created_vars) == 0


def next_vars_to_create(fdict, created_vars):
    """
    Inform the dependent variables that already have the
    necessary variables to be created.

    Parameters
    ----------
    fdict : dict of sets of str
        The keys are the names of dependent variables and the dict 
        values are the set of names of variables on which the 
        key depends on.
    created_vars : set of str
        Set of names of already created variables.
    
    Returns
    -------
    new_vars : set of str
        Names of the variables that are ready to be created.
    """
    
    new_vars = set()
    for dep_var in fdict:
        if ready_to_create(fdict[dep_var], created_vars):
            new_vars.add(dep_var)
    
    return new_vars


def depvar_creation_order(fdict, sources):
    """
    List the order in which to create dependent variables so
    the required input variables are available for each 
    dependent variable.
    
    Parameters
    ----------
    fdict : dict of sets of str
        The keys are the names of dependent variables and the dict 
        values are the set of names of variables on which the 
        key depends on.
    sources : set of str
        Set of names of the independent variables (sources).
    
    Returns
    -------
    creation_order : list of str
        Order in which to create the dependent variables so to 
        guarantee that the necessary variables are already in place.
    """
    
    creation_order = []
    
    created_vars = sources.copy()
    depf = fdict.copy()
    while len(depf) > 0:
        # Get next variables to create:
        next_vars = next_vars_to_create(depf, created_vars)
        assert len(next_vars) > 0, 'Cannot create any variables given the existent ones.'
        # Add it to creation order and to created variables:
        creation_order += sorted(list(next_vars))
        created_vars.update(next_vars)
        # Remove created variables from dependency listing (no need to check again):
        for k in next_vars:
            depf.pop(k)
            
    return creation_order


def generate_causal_data(causal_links, n_samples=10000):
    """
    Generate random data that obeys the specified causal links.
    
    Parameters
    ----------
    causal_links : list of tuples of str
        Each tuple (x, y) represents a causal relation x -> y.
        x and y (str) are names of variables.
    n_samples : int
        Number of instances to generate.
    
    Returns
    -------
    df: DataFrame
        Mock data that obeys the specified causal links.
    """
    
    # Start DataFrame:
    df = pd.DataFrame()

    # Get set of sources:
    sources = links2sources(causal_links)
    
    # Generate source variables:
    for x in sorted(list(sources)):
        df[x] = generate_source(n_samples=n_samples)

    # Get variables' dependencies:
    dependencies = links2funcdict(causal_links)
    creation_order = depvar_creation_order(dependencies, sources)

    # Generate dependent variables:
    for y in creation_order:
        y_src = sorted(list(dependencies[y]))
        X = [df[s].values for s in y_src]
        df[y] = generate_dependent(X)
        
    return df
