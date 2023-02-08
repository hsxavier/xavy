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