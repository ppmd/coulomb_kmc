__author__ = "W.R.Saunders"


import ctypes
import numpy as np
from math import *

from functools import lru_cache

REAL = ctypes.c_double
INT64 = ctypes.c_int64

from coulomb_kmc.common import PROFILE, ProfInc


class LocalOctalBase(ProfInc):
    """
    Base class that is inherited by local and octal classes.
    Provides common methods, such as cell linear index to tuple index mapping.
    """

    def _get_fmm_cell(self, ix, cell_map, slow_to_fast=False):
        # produces xyz tuple by default
        cc = cell_map[ix][0]
        cx, cy, cz = self._cell_lin_to_tuple(cc)

        if not slow_to_fast:
            return cx, cy, cz
        else:
            return cz, cy, cx
 
    def _cell_lin_to_tuple_no_check(self, cc):

        R = self.fmm.R
        sl = 2 ** (R - 1)
        cx = cc % sl
        cycz = (cc - cx) // sl
        cy = cycz % sl
        cz = (cycz - cy) // sl
        return cx, cy, cz

    def _cell_lin_to_tuple(self, cc):

        cx, cy, cz = self._cell_lin_to_tuple_no_check(cc)
        els = self.entry_local_size
        elo = self.entry_local_offset

        assert cz >= elo[0] and cz < elo[0] + els[0]
        assert cy >= elo[1] and cy < elo[1] + els[1]
        assert cx >= elo[2] and cx < elo[2] + els[2]
        return cx, cy, cz


    def _global_cell_xyz(self, tcx):
        """get global cell index from xyz tuple"""
        gcs = self.global_cell_size
        return tcx[0] + gcs[0] * ( tcx[1] + gcs[1] * tcx[2] )
