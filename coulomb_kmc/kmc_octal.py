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
        # in future domains may not be square
        csc = [csc, csc, csc]
        csw = [self.domain.extent[0] / csc[0],
               self.domain.extent[1] / csc[1],
               self.domain.extent[2] / csc[2]]
        
        # this is pad per dimension
        pad = [1 + int(ceil(max_move/cx)) for cx in csw]
 
        ls = fmm.tree.entry_map.local_size
        lo = fmm.tree.entry_map.local_offset

        # as offset indices
        pad_low = [list(range(-px, 0)) for px in pad]
        pad_high = [list(range(lsx, lsx + px)) for px, lsx in zip(pad, reversed(ls))]

        
        print("ls", ls, "lo", lo, "extent", self.domain.extent, "boundary", self.domain.boundary)
        
        # cell indices as offsets from owned octal cells
        cell_indices = [ lpx + list(range(lsx)) + hpx for lpx, lsx, hpx in zip(pad_low, reversed(ls), pad_high) ]
        cell_indices = [[ (cx + osx) % cscx for cx in dx ] for dx, cscx, osx in zip(cell_indices, csc, reversed(lo))]
        
        print(cell_indices)
        
        



