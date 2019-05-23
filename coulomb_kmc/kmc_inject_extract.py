"""
Module to handle injection and extraction of charges from a PyFMM instance.

Currently implemented for 1 MPI rank only.
"""

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
        self._bc = kmcfmm._bc
        self.fmm = kmcfmm.fmm
        self.domain = kmcfmm.domain
        L = self.fmm.L
        self.L = L
        self.ncomp = 2*(L**2)
        self.half_ncomp = L**2
        
        self._lee = LocalExpEval(self.L)
        self._lrc = LongRangeMTL(L, self.domain)

        e = self.domain.extent
        assert abs(e[0] - e[1]) < 10.**-15
        assert abs(e[0] - e[2]) < 10.**-15
        if self._bc == BCType.FREE_SPACE:
            self._direct = kmc_direct.FreeSpaceDirect()
        elif self._bc is BCType.NEAREST:
            self._direct = kmc_direct.NearestDirect(float(e[0]))
        elif self._bc is BCType.PBC:
            self._direct = kmc_direct.PBCDirect(float(e[0]), self.domain, L)
        else:
            raise NotImplementedError('BCType unknown: ' + str(self._bc))


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


    def propose_extract(self, ids):
        """
        Propose the extraction of a set of charges by providing the local
        particle ids. Returns the change of energy if the charges were
        removed.

        :arg ids: Iterable of local charge id to remove.
        """
        
        assert self.kmcfmm.comm.size == 1

        # code is written assuming the current state is A + B for A, B sets
        # of charges. B is the set to remove. Hence energy is formed of the
        # AA + AB + BB interactions.

        ids = np.array(ids, dtype=INT64)

        # BB interations (required to avoid double count of BB)
        BB_energy = self.compute_energy(
            self.kmcfmm.positions[ids, :], self.kmcfmm.charges[ids, :])

        # AB + BB interactions
        AB_BB_energy = 0.0
        for ix in ids:
            ix = int(ix)
            AB_BB_energy += self.kmcfmm._charge_indirect_energy_old(ix) + \
                self.kmcfmm._direct_contrib_old(ix)
        
        if self._bc == BCType.PBC:
            tmp_field = np.zeros(len(ids), REAL)
            self.kmcfmm._lr_energy.eval_field(
                self.kmcfmm.positions.view[ids, :], tmp_field)

            for ixi, ix in enumerate(ids):
                tmp_field[ixi] *= self.kmcfmm.charges.view[ix, 0]

            AB_BB_LR_energy = np.sum(tmp_field)

        else:
            AB_BB_LR_energy = 0.0

        return -1.0 * AB_BB_energy + BB_energy - AB_BB_LR_energy


    def propose_inject(self, positions, charges):
        """
        Propose the injection of a set of charges. Returns the change in system
        energy if the set of charges were added.

        :arg positions: New charge positions.
        :arg charges: New charge values.
        """

        assert self.kmcfmm.comm.size == 1

        N = positions.shape[0]
        BB_energy = self.compute_energy(positions, charges)
        field_values = self.kmcfmm.eval_field(positions).reshape(N)
        AB_energy = float(np.sum(np.multiply(charges.reshape(N), field_values)))

        return AB_energy + BB_energy


    def extract(self, ids):
        """
        Extract the set of charges given by local ids.

        :arg ids: Iterable of particle local ids to remove.
        """

        assert self.kmcfmm.comm.size == 1

        with self.kmcfmm.group.modify() as m:
            if ids is not None:
                m.remove(ids)

        self.kmcfmm.initialise()


    def inject(self, add):
        """
        Inject a set of charges. e.g.

        ::
            IE = InjectorExtractor(....)
            IE.inject({
                A.P: np.array((
                    (r_x1, r_y1, r_z1),
                    (r_x2, r_y2, r_z2),
                )),
                A.Q: np.array(((-1.0), (1.0))),
                ...
            })



        :arg add: Dictonary of the style of `state.modifier.add`.
        """

        assert self.kmcfmm.comm.size == 1
        with self.kmcfmm.group.modify() as m:
            if add is not None:
                m.add(add)

        self.kmcfmm.initialise()





























