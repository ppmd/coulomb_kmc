from __future__ import division, print_function, absolute_import
__author__ = "W.R.Saunders"

"""
Octal Tree classes for kmc
"""

import numpy as np
from math import ceil

class LocalCellExpansions(object):
    """
    Object to get, store and update local expansions from an fmm instance.
    """
    def __init__(self, fmm, max_move):
        self.fmm = fmm
        self.max_move = max_move
        self.domain = fmm.domain
        
        csc = fmm.tree.entry_map.cube_side_count
        csw = [self.domain.extent[0] / csc,
               self.domain.extent[1] / csc,
               self.domain.extent[2] / csc]
        
        pad = [1 + int(ceil(max_move/cx)) for cx in csw]

        
        



