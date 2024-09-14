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