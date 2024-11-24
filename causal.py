#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Functions for causal inference and causal diagrams.
Copyright (C) 2024  Henrique S. Xavier
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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

###############################################################################################
### Class of function describing the causal relationship between a variable and its parents ###
###############################################################################################

class Function():
    """
    A function describing the relationship between the variable in question 
    `y` and its parents `X`. It is composed of two parts: the expected, 
    deterministic value E, and an additive noise N:

    y(X) = E(X) + N(X)

    Parameters
    ----------
    parents : iterable
        A list or set of variable names that represent its parents.
    type : str
        The type of deterministic relation between the variable and its
        parents. Currently, the options are:
        - 'linear': A linear relationship;
        - 'log_linear': A shifted log-linear relationship, where the 
          logarithms of the parents are linearly combined, and the variable 
          is an exponential of the parents.
        - 'custom': Any function specified by a formula provided by the user.
    noise_type : str
        Property of the random noise added to the variable. Currently, it 
        can be:
        - 'gaussian': A homoscedastic, zero-mean, additive noise drawn from 
          a Normal distribution.
        - 'lognormal': A heteroscedastic, zero-mean, additive noise whose
          amplitude is proportional to the unshifted deterministic value of 
          the variable and the distribution is lognormal. This is analogous
          to a gaussian noise added to the exponent of the deterministic 
          log-linear function.
        - 'multinomial': A random sampling from a specified set of possible 
          values in which the probabilities of each possible value is 
          given.    
    """
    
    def __init__(self, parents, type=None, noise_type=None):
        
        self.parents = list(parents)
        self.type = type
        self.noise_type = noise_type

    def set_linear(self, coefs_dict, intercept=0, log_linear=False, min_value=0):
        """
        Make the Function 'linear' or 'log-linear' and set its parameters.

        Parameters
        ----------
        coefs_dict : dict
            Dict from parents names (str) to their linear coefficients
            (float).
        intercept : float
            Constant value in the linear relationship between the 
            variable and its parents. If the relationship is log-linear,
            this is a constant term in the exponent of the log-linear 
            equation, which is equivalent to a factor on the variable.
        log_linear : bool
            If True, the relation set between variable and parents is 
            log-linear instead of linear.
        min_value : float
            This is the mininum value of the variable in case the 
            relation is log-linear.
        """
        
        # Set provided parameters:
        self.coefs_dict = coefs_dict
        self.coefs      = np.array([coefs_dict[k] for k in self.parents])
        self.intercept  = intercept

        # Specific for log-linear:
        if log_linear is True:
            self.type = 'log_linear'
            self.min_value = min_value
        # Specific for linear:
        else:
            self.type = 'linear'
            # Clear old attributes:
            if hasattr(self, 'min_value'):
                del self.min_value

        # Clear old attributes:
        if hasattr(self, 'f'):
            del self.f
    
    def set_custom(self, func):
        """
        Set the relationship between the variable and its parents
        as an arbitrary function provided by the user.

        Parameters
        ----------
        func: callable
            The callable takes as input the values of the 
            variable's parents (the function's parameters names 
            must be the parents names specified when instantiating 
            the Function) and returns the value of the variable.
            The relationship is expected to be deterministic (so the
            specified noise is added to it), but `func` may be 
            stochastic.
        """
        
        # Set provided parameters:
        self.type = 'custom'
        self.f = func

        # Clear old atttributes:
        if hasattr(self, 'coefs_dict'):
            del self.coefs_dict
        if hasattr(self, 'coefs'):
            del self.coefs
        if hasattr(self, 'intercept'):
            del self.intercept
        if hasattr(self, 'min_value'):
            del self.min_value

    def _parse_parents_values(self, parents_values=None, **parents_values_kwargs):
        """
        Standardize the function's arguments as an array of floats.
        The order in the array is the same as specified in 
        `parents` attribute during instantiation.

        Parameters
        ----------
        parents_values : array (2D)
            Array with the parents values already in the appropriate
            order. This is passed as output. First dimension (rows)
            are the parents and second dimension (columns) are the 
            realizations.
        parents_values_kwargs : dict
            The keys are parents names (str) and the values are 
            arrays of floats, each entry being a realization.

        Returns
        -------
        parents_values : array (2D)
            Either the input `parents_values` or the `parents_values_kwargs`
            parsed to an array.
        """
        if self.type == None:
            raise ValueError('Function has not been set.')
        
        # Make sure inputs are specified only once:
        if type(parents_values) != type(None):
            assert len(parents_values_kwargs) == 0, '`One cannot pass both `parents_values` and `**parents_values_kwargs`.'
                
        # Parse input into array:
        elif len(parents_values_kwargs) > 0:
            parents_values = np.array([parents_values_kwargs[k] for k in self.parents])

        return parents_values

    def compute(self, parents_values=None, **parents_values_kwargs):
        """
        Compute the deterministic value of the variable given its parents.

        Parameters
        ----------
        parents_values : array (2D)
            Array with the parents values already in the appropriate
            order (as specified in the `parents` attribute). First dimension 
            (rows) are the parents and second dimension (columns) are the 
            realizations.
        parents_values_kwargs : dict
            The keys are parents names (str) and the values are 
            arrays of floats, each entry being a realization.

        Returns
        -------
        value : array
            The deterministic values of the variable in each realization.
        """
        
        # Parse input: 
        parents_values = self._parse_parents_values(parents_values, **parents_values_kwargs)        
        
        # Compute the linear formula:
        if self.type == 'linear':
            # Expecting coefs as a 1D array and parents_values as 2D (n_variables, n_instances):
            scalar_prod = self.coefs.dot(parents_values)
            # For generating sources:
            if len(scalar_prod) == 0:
                scalar_prod = 0
            
            return scalar_prod + self.intercept

        # Compute the log-linear formula:
        elif self.type == 'log_linear':
            # Expecting coefs as a 1D array and parents_values as 2D (n_variables, n_instances):
            scalar_prod = self.coefs.dot(parents_values)
            # For generating sources:
            if len(scalar_prod) == 0:
                scalar_prod = 0
            
            return np.exp(scalar_prod + self.intercept) + self.min_value
            
        # Compute custom formula:
        elif self.type == 'custom':
            assert len(parents_values_kwargs) > 0, "Current implementation of 'custom' functions only allows for `parents_values_kwargs` arguments, not `parents_values`."
            return self.func(**parents_values_kwargs)

    def set_noise(self, type=None, scale=1.0, probs_dict={0:0.5, 1:0.5}):
        """
        Set parameters for the variable's noise. 

        Parameters
        ----------
        type : str or None
            The type of error. Options are:
            - 'gaussian': A homoscedastic, zero-mean, additive noise drawn from 
              a Normal distribution.
            - 'lognormal': A heteroscedastic, zero-mean, additive noise whose
              amplitude is proportional to the unshifted deterministic value of 
              the variable and the distribution is lognormal. This is analogous
              to a gaussian noise added to the exponent of the deterministic 
              log-linear function.
            - 'multinomial': A random sampling from a specified set of possible 
              values in which the probabilities of each possible value is 
              given.
            If unspecified (i.e. None), use the type specified when instantiating
            the Function.
        scale : float
            Standard deviation of the 'gaussian' or 'lognormal' noise.
            In the latter type, it is the standard deviation for deterministic 
            variable value equal to 1. I believe it also boils down to a 
            multiplicative fractional error scale.
        probs_dict : dict
            Dict from possible discrete noise values to their probabilities.
            Only used for 'multinomial' noise type.
        """

        if type == None:
            # If not passed, use previously specified noise type: 
            type = self.noise_type
        else:    
            # Save noise type:
            self.noise_type  = type

        if type in {'gaussian', 'lognormal'}:
            # Set attributes:
            self.noise_scale = scale
            # Clear old attributes:
            if hasattr(self, 'noise_probs'):
                del self.noise_probs                
           
        elif type == 'multinomial':
            # Security check:
            total_prob = np.array(list(probs_dict.values())).sum()
            assert np.isclose(total_prob, 1.0), "Probabilites in `probs_dict` must add to 1."
            # Set attributes:
            self.noise_probs = probs_dict
            # Clear old attributes:
            if hasattr(self, 'noise_scale'):
                del self.noise_scale
            
    def random_init(self, type=None, noise_type=None):
        """
        Randomly sets the parameters of the deterministic relationship
        and independent noise for the variable.

        Parameters
        ----------
        type : str
            The type of deterministic relation between the variable and its
            parents. Currently, the options are:
            - 'linear': A linear relationship;
            - 'log_linear': A shifted log-linear relationship, where the 
              logarithms of the parents are linearly combined, and the variable 
              is an exponential of the parents.
            - 'custom': Any function specified by a formula provided by the user.
            If unspecified (i.e. None), use the type specified when instantiating
            the Function.
        noise_type : str or None
            The type of error. Options are:
            - 'gaussian': A homoscedastic, zero-mean, additive noise drawn from 
              a Normal distribution.
            - 'lognormal': A heteroscedastic, zero-mean, additive noise whose
              amplitude is proportional to the unshifted deterministic value of 
              the variable and the distribution is lognormal. This is analogous
              to a gaussian noise added to the exponent of the deterministic 
              log-linear function.
            - 'multinomial': A random sampling from a specified set of possible 
              values in which the probabilities of each possible value is 
              given.
            If unspecified (i.e. None), use the type specified when instantiating
            the Function.        
        """

        # Set relationship type:
        if type == None:
            # If not passed, use previously specified noise type: 
            type = self.type
        else:    
            # Save noise type:
            self.type  = type        

        # Set noise type:
        if noise_type == None:
            # If not passed, use previously specified noise type: 
            noise_type = self.noise_type
        else:    
            # Save noise type:
            self.noise_type  = noise_type 
            
        # Set mean linear trend:
        if type == 'linear':
            coefs_dict = {k: -2 + 4 * np.random.random() for k in self.parents}
            intercept  = -10 + 20 * np.random.random()
            self.set_linear(coefs_dict, intercept)
        elif type == 'log_linear':
            coefs_dict = {k: -2 + 4 * np.random.random() for k in self.parents}
            intercept  = -4 + 8 * np.random.random()
            min_value  = -10 + 20 * np.random.random()
            self.set_linear(coefs_dict, intercept, log_linear=True, min_value=min_value)
        else:
            raise ValueError("Unknown type '{:}'.".format(type))
        
        # Set noise parameters:
        if noise_type == 'gaussian':
            scale = 0.01 + 5.99 * np.random.random()
            self.set_noise(noise_type, scale)
        elif noise_type == 'lognormal':
            scale = 0.01 + 0.99 * np.random.random()
            self.set_noise(noise_type, scale)
        elif noise_type == 'multinomial':
            p1 = np.random.random()
            p0 = 1 - p1
            probs_dict = {0:p0, 1:p1}
            self.set_noise(noise_type, probs_dict=probs_dict)            
        else:
            raise ValueError("Unknown noise type '{:}'.".format(noise_type))

    
    def gen_noise(self, parents_values=None, n_samples=None, **parents_values_kwargs):
        """
        Generate noise for the variable given its parents values (in case of 
        heteroscedastic noise) or number of realizations. This can provide the 
        value of source variables as well (since they are only noise).

        Parameters
        ----------
        parents_values : array (2D)
            Array with the parents values already in the appropriate
            order (as specified in the `parents` attribute). First dimension 
            (rows) are the parents and second dimension (columns) are the 
            realizations. 
        n_samples : int
            Number of realizations to generate, if `parents_values` or 
            `parents_values_kwargs` are not provided.
        parents_values_kwargs : dict
            The keys are parents names (str) and the values are 
            arrays of floats, each entry being a realization.

        Returns
        -------
        noise : array
            The realizations of the noise for the variable. The number 
            of realizations (and the size of the array) is determined
            by `n_samples` or the length of realizations in `parents_values` 
            or `parents_values_kwargs`.
        """
        
        if type(self.noise_type) == type(None):
            raise ValueError('Noise has not been set.')

        # Parse input: 
        parents_values = self._parse_parents_values(parents_values, **parents_values_kwargs)        

        if self.noise_type == 'gaussian':
            
            # Guard against conflicting input:
            if type(parents_values) != type(None) and type(n_samples) != type(None):
                if parents_values.shape[1] != n_samples:
                    raise ValueError('`n_samples` should be equal to number of instances in `parents_values`.')
            if n_samples == None:
                n_samples = parents_values.shape[-1]
                
            return np.random.normal(scale=self.noise_scale, size=n_samples)

        elif self.noise_type == 'lognormal':

            # Guard against conflicting input:
            if type(parents_values) != type(None) and type(n_samples) != type(None):
                if parents_values.shape[1] != n_samples:
                    raise ValueError('`n_samples` should be equal to number of instances in `parents_values`.')
            if n_samples == None:
                n_samples = parents_values.shape[-1]

            # Compute noise scale (prop. to the variable's value):
            if self.type == 'linear':
                ns = self.compute(parents_values)
            elif self.type == 'log_linear':
                ns = self.compute(parents_values) - self.min_value
                                        
            # Compute parameters for associated Gaussian variable:
            w   = np.log(1 + self.noise_scale ** 2)
            mu  = -w / 2
            sig = np.sqrt(w)

            # Generate associated Gaussian noise:
            g = np.random.normal(loc=mu, scale=sig, size=n_samples)

            return ns * (np.exp(g) - 1) 
        
        elif self.noise_type == 'multinomial':
            assert type(n_samples) != type(None), "Please provide `n_samples` for 'multinomial' noise."
            values = list(self.noise_probs.keys())
            probs  = list(self.noise_probs.values())
            return np.random.choice(np.array(values, dtype=int), size=n_samples, p=probs)

        else:
            raise ValueError("Unknown noise type '{:}'".format(self.noise_type))
    
    def generate(self, parents_values=None, n_samples=None, **parents_values_kwargs):
        """
        Generate realizations for the varible given its parents values, 
        including both the deterministic value and noise.

        Parameters
        ----------
        parents_values : array (2D)
            Array with the parents values already in the appropriate
            order (as specified in the `parents` attribute). First dimension 
            (rows) are the parents and second dimension (columns) are the 
            realizations. 
        n_samples : int
            Number of realizations to generate, if `parents_values` or 
            `parents_values_kwargs` are not provided. Useful for source
            variables.
        parents_values_kwargs : dict
            The keys are parents names (str) and the values are 
            arrays of floats, each entry being a realization.

        Returns
        -------
        data : array
            The realizations of the variable, including noise. The number 
            of realizations (and the size of the array) is determined
            by `n_samples` or the length of realizations in `parents_values` 
            or `parents_values_kwargs`.
        """
        
        # Parse input: 
        parents_values = self._parse_parents_values(parents_values, **parents_values_kwargs)        
        
        if self.type in {'linear', 'log_linear', 'custom'}:
            mean  = self.compute(parents_values)
            noise = self.gen_noise(parents_values, n_samples)
            return mean + noise

        
################################
### Class for a causal model ###
################################

class CausalModel():
    """
    A class for storing and simulating data generated through a
    causal model specified by a causal graph and generating 
    functions.

    Parameters
    ----------
    edges : list of tuple of str
        These are the causal links between variables. Each tuple
        represents a directed edge in the causal graph and 
        contain two strings, each one identifying a variable: 
        the first one is the parent node and the last one is the
        child node. Each tuple (x, y) represents a causal relation 
        x -> y. x and y (str) are names of variables.

    Attributes
    ----------
    variables : list
        List of variables (nodes) in the graph, in descending order, 
        from ancestors to descendants.
    sources : set
        Variables (nodes) that have no parents.
    parents : dict of sets
        For each variable identifier (str) as key, the dict 
        returns a set of identifiers (str) for the variables' 
        parents.
    dependent : list
        List of variables that depend on others (i.e. have parents).
    functions : dict of Functions
        For each variable identifier (str) as key, the dict returns 
        a Function object that specifies how the value of the 
        respective variable is set given its parents.
    """
    
    def __init__(self, edges=[]):
        self.edges = edges
        #self.variables = self._links2varset()
        self.sources   = self._links2sources()
        self.parents   = self._links2funcdict()
        self.dependent = self._depvar_creation_order()
        self.variables = sorted(list(self.sources)) + self.dependent
        self.functions = {v:Function(self.parents[v]) for v in self.variables}
    
    def _links2varset(self, only_dep=False):
        """
        Get the set of variables present in the causal links (edges)
        of the Graph.
        
        Parameters
        ----------
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
    
        variables = set()
        for x, y in self.edges:
            if only_dep is False:
                variables.add(x)
            variables.add(y)
        
        return variables
    
    def _links2funcdict(self):
        """
        Identify variables' dependencies given the edges.
            
        Returns
        -------
        fdict : dict of set of str
            The keys are the names of variables and the dict 
            values are the set of names of variables on which the 
            key depends on. Sources have empty sets associated to 
            them.
        """
        
        fdict = defaultdict(lambda: set())
        for link in self.edges:
            # Get sources `x` and dependent variables `y`:
            x = link[0]
            y = link[1]
            # Add `x` to `y` variables list:
            fdict[y].add(x)

        for source in self.sources:
            fdict[source] = set()
        
        return dict(fdict)

    def _links2sources(self):
        """
        Find the set of independent variables (sources) in
        the list of causal links (edges of the CausalModel).
        
        Returns
        -------
        src_set : set of str
            The set of names of independent variables (str) present in 
            `causal_links`.
        """
        all_set = self._links2varset()
        dep_set = self._links2varset(True)
        src_set = all_set - dep_set
        
        return src_set

    def _ready_to_create(self, source_list, created_vars):
        """
        Inform whether all variables in `source_list` (set) already 
        have been created (i.e. are in set `created_vars`) or not.
        """
        
        return len(source_list - created_vars) == 0
    
    def _next_vars_to_create(self, fdict, created_vars):
        """
        Inform the dependent variables that already have the
        necessary variables to be created.
    
        Parameters
        ----------
        fdict : dict of sets of str
            The keys are the names of variables and the dict 
            values are the set of names of variables on which the 
            key depends on. Sources (parentless variables) have 
            empty sets.
        created_vars : set of str
            Set of names of already created variables.
        
        Returns
        -------
        new_vars : set of str
            Names of the variables that are ready to be created.
        """
        
        new_vars = set()
        for dep_var in fdict:
            if self._ready_to_create(fdict[dep_var], created_vars):
                new_vars.add(dep_var)
        
        return new_vars
    
    def _depvar_creation_order(self):
        """
        List the order in which to create dependent variables so
        the required input variables are available for each 
        dependent variable.

        Returns
        -------
        creation_order : list of str
            Order in which to create the dependent variables so to 
            guarantee that the necessary variables are already in place.
        """
        
        creation_order = []

        # Create copy of graph data (these will be modified in the processing below):
        created_vars = self.sources.copy()
        depf = self.parents.copy()
        # Remove sources from keys in dict of parents:
        for s in self.sources:
            depf.pop(s)
        
        while len(depf) > 0:
            # Get next variables to create:
            next_vars = self._next_vars_to_create(depf, created_vars)
            assert len(next_vars) > 0, 'Cannot create any variables given the existent ones.'
            # Add it to creation order and to created variables:
            creation_order += sorted(list(next_vars))
            created_vars.update(next_vars)
            # Remove created variables from dependency listing (no need to check again):
            for k in next_vars:
                depf.pop(k)

        return creation_order

    def _format_node(self, G, node_name, node_type, node_format={'target':'filled', 'hidden':'dashed'}):
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

    def plot_diagram(self, special_nodes=None, rankdir='LR'):
        """
        Given the specified list of directed edges (represented
        by tuples of two elements), draws a graph.
    
        Parameters
        ----------
        special_nodes : dict
            Two keys are allowed in the dict: 'hidden' and 'target'.
            The value of each key is a list of str containing names
            of nodes to be marked as such. Hidden have dashed contours,
            target have gray filling.
        """
        
        # Build graph:
        G = gv.AGraph(directed=True, rankdir=rankdir)
        G.add_edges_from(self.edges)
        
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
                    self._format_node(G, node_name, node_type)
        
        # Draw graph:
        G.layout(prog='dot')
        img_bytes = G.draw(format='png')
        image = Image.open(io.BytesIO(img_bytes))
        display(image)

    def random_init(self, type='linear', noise_type='gaussian'):
        """
        Randomly sets the functions that specify the values of 
        variables given their parents. It also specifies how 
        noise is added to the variables.

        Parameters
        ----------
        type : str
            What kind of function is used, among the options:
            - 'linear': a linear combination of its parents, 
              including an intercept, plus additive noise.
        noise_type : str
            What kind of noise to be added to the variable, 
            among the options:
            - 'gaussian': normal distribution of fixed-scale.
            (homoscedastic errors).
        """
        for v in self.variables:
            self.functions[v].random_init(type, noise_type)

    def generate_data(self, n_samples=100):
        """
        Generate data for all the variables in the CausalModel
        according to their causal relations and the functions 
        that set their values. Data is saved in the object as 
        a dict from variable identifiers to numpy arrays 
        containing the data.

        Parameters
        ----------
        n_samples : int
            Number of independent realizations of the variable set
            in the CausalModel.

        Returns
        -------
        df : DataFrame
            Table containing the realizations (rows) for the 
            variables (columns).
        """
        
        # Check if every function is set:
        for v in self.variables:
            assert type(self.functions[v].type) != type(None), "Function for variable '{:}' is not set."
        
        # Generate sources:
        self.data = dict()
        for v in self.sources:
            self.data[v] = self.functions[v].generate(n_samples=n_samples)
        # Generate dependent variables:
        for v in self.dependent:
            parents_data = {p:self.data[p] for p in self.parents[v]}
            self.data[v] = self.functions[v].generate(**parents_data)

        return pd.DataFrame(self.data)


################################
### Measuring causal effects ###
################################

def compute_dependency(list_of_variables, y, n_trials=40):
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
    assert len(X) == len(y), '`X` and `y` must have the same length.'
    n     = len(X)
    idx   = np.arange(n, dtype=int)
    
    model = LinearRegression()

    # LOOP over fit trials:
    coefs = []
    for _ in range(n_trials):
        # Create bootstrap sample:
        si = np.random.choice(idx, size=n, replace=True)
        train_X = X[si]
        train_y = y[si]
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


def gen_intervention_data(df, do_values):
    """
    Create a synthetic dataset in which every instance is 
    replicated for each specified value of the do variables.

    Parameters
    ----------
    df : DataFrame
        Original, real dataset.
    do_values : dict of list-likes
        Keys are the variables we will intervene on. The values
        are like of values we will force them to assume.

    Returns
    -------
    out_df : DataFrame
        The cross join between the specified values in `do_values` 
        and the remaining features in the dataset.
    """

    out_df = df.copy()
    
    # Loop over do-variables:
    for k in do_values.keys():

        # Create 'intervention' data:
        v = do_values[k]
        v_series = pd.Series(v).repeat(len(df))
        v_series.index = np.concatenate([df.index] * len(v))
        v_series.name = k
    
        # Replace do variable with 'intervention' values:
        out_df = out_df.drop(k, axis=1)
        out_df = out_df.join(v_series)

    return out_df


def adjustment_predict_intervention(model, X_df, do_var, do_values):
    """
    Predict the effect of intervening on a variable by setting
    it to the specified values, using the adjustment formula.

    Parameters
    ----------
    model : sklearn predictor
        A model for the target variable already fitted and 
        ready to predict. It must expect as input the do 
        feature plus all features whose conditioning is 
        required to estimate the causal effect of the do
        variable.
    X_df : DataFrame
        The independent variables (features) used in the 
        prediction. It must include the do variable plus all
        features we are conditioning on. The instances must 
        follow the true distribution of the data in the feature
        space.
    do_var : str
        Name of the variable to intervene on. It must 
        be one of the features in `X_df`.
    do_values : list-like
        Do variable values for which to evaluate the 
        target value.

    Returns
    -------
    marg_y_preds : Series
        Estimate of the average causal effect of the do 
        variable on the target variable.
    """
    
    # Generate synthetic data:
    doX_df = gen_intervention_data(X_df, {do_var: do_values})
    # Predict for all data:
    all_y_preds = model.predict(doX_df)
    # Take the average for each do value:
    doX = doX_df[do_var].values
    marg_y_preds = pd.Series(all_y_preds).groupby(doX).mean()

    return marg_y_preds 


def measure_causal(model, df, do_x, x_values, target_y, cond_z):
    """
    Measure the causal effect of `x` on `y` using the backdoor
    adjustment formula, i.e., by conditioning on `z`, making 
    predictions, and integrating over `z` to compute the average 
    effect of `x`.

    Parameters
    ----------
    model : sklearn Predictor
        An adequate predictor for `y` given `x` and `z`. 
        Hyperparameters are expected to be set to the 
        appropriate values. The model does not need to be 
        fit.
    df : DataFrame
        Dataset containing the target `y`, the variable `x` where
        we would like to intervene, and the remaining `z` variables
        required to block backdoor paths from `x` to `y`.
    do_x : str
        Name of the `x` variable as specified in `df` header.
    x_values : list-like
        Values of `x` for which to compute the value of `y` as if 
        we had intervene on `x`.
    target_y : str
        Name of the target `y` variable as specified in the `df` 
        header.
    cond_z : list of str
        List of names of variables (as specified in `df` header)
        that block all backdoor (i.e. non-causal) paths from `x` 
        to `y`.

    Returns
    -------
    y_pred_final : array
        Prediction of the average value of `y` had we intervened 
        on `x`.
    """
    
    # Initialize KFold with 5 splits:
    kf = KFold(n_splits=5, shuffle=True, random_state=42111)
    
    # Loop over data splits:
    y_preds = []
    for train_index, test_index in kf.split(df):
        print('Test set size: ', len(test_index)) # sets must have approx. the same size.
        
        # Train model:
        train_df = df.iloc[train_index]
        X_train  = train_df[cond_z + [do_x]]
        y_train  = train_df[target_y]
        model.fit(X_train, y_train)
        
        # Predict:
        test_df = df.iloc[test_index]
        X_test = test_df[cond_z + [do_x]] 
        y_pred = adjustment_predict_intervention(model, X_test, do_x, x_values)
        y_preds.append(y_pred)

    # Combine results:
    y_pred_series = pd.concat(y_preds)
    y_pred_final  = y_pred_series.groupby(y_pred_series.index).mean()

    return y_pred_final
