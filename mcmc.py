#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Auxiliary code for pyMC
Copyright (C) 2024  Henrique S. Xavier
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

import numpy as np


def get_chains(idata, par_name):
    """
    Extract chains from pyMC `idata` to a numpy array.
    """
    # Get data specs:
    n_chains = idata['posterior'].dims['chain']
    n_draws  = idata['posterior'].dims['draw']

    # Get sampling parameters:
    post_dims  = idata['posterior'].dims
    # Get data dimension levels:
    dim_levels = list(post_dims)[2:] # Drop chain and draw indices.
    # Get data dimensions:
    data_dims = [post_dims[k] for k in dim_levels]
    
    # Stack parallel chains into a single sampling set:
    dims_stacked = [n_chains * n_draws] + data_dims
    stacked_chains = np.array(idata['posterior'][par_name]).reshape(dims_stacked)

    return stacked_chains


# If running this code as a script:
if __name__ == '__main__':
    pass
