__author__ = "W.R.Saunders"


import ctypes
import numpy as np
from math import *
import time

REAL = ctypes.c_double
INT64 = ctypes.c_int64

# cuda imports if possible
import ppmd
import ppmd.cuda
from ppmd.coulomb.sph_harm import *
from ppmd.lib.build import simple_lib_creator, LOADED_LIBS
from coulomb_kmc.common import spherical, cell_offsets_26, ProfInc, BCType
from ppmd.coulomb.fmm_pbc import LongRangeMTL
from coulomb_kmc.kmc_expansion_tools import LocalExpEval
from coulomb_kmc import kmc_direct


class InjectorExtractor(ProfInc):
    """
    Class to propose and accept the injection and extraction of charges.

    :arg kmcfmm: PyFMM instance.
    """

    def __init__(self, kmcfmm):
        
        self.kmcfmm = kmcfmm
        self._bc = kmcfmm.bc
        self.fmm = kmcfmm.fmm
        self.domain = kmc.domain
        L = self.fmm.L
        self.L = L
        self.ncomp = 2*(L**2)
        self.half_ncomp = L**2
        
        self._lee = LocalExpEval(self.L)
        self._lrc = LongRangeMTL(L, self.domain)


        e = self.domain.extent[0]
        assert abs(e[0] - e[1]) < 10.**-15
        assert abs(e[0] - e[2]) < 10.**-15
        if self._bc == BCType.NEAREST:
            self._direct = kmc_direct.FreeSpaceDirect()
        elif self._bc is BCType.NEAREST:
            self._direct = kmc_direct.NearestDirect(float(e[0]))
        elif self._bc is BCType.PBC:
            self._direct = kmc_direct.PBCDirect(e, self.domain, L)
        else:
            raise NotImplementedError('BCType unknown')


    def compute_energy(self, positions, charges):
        """
        Compute the energy of a set of charges in a domain.
        Assumes that the number of charges is small.

        :arg positions: Nx3, c_double NumPy array of positiions.
        :arg charges: Nx1, c_double NumPy array of charge values.
        """
    
        positions = np.atleast_2d(positions).copy()
        N = positions.shape[0]
        assert positions.shape[1] == 3

        charges = np.atleast_2d(charges).copy()
        assert charges.shape[0] == N
        assert charges.shape[1] == 1

        assert positions.dtype == REAL
        assert charges.dtype == REAL

        phi = self._direct(N, positions, charges)

        return phi







