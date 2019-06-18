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

from ppmd.access import *

class InjectorExtractor(ProfInc):
    """
    Class to propose and accept the injection and extraction of charges. 
    Is inherited by KMCFMM.
    """

    def __init__(self):
        
        e = self.domain.extent
        assert abs(e[0] - e[1]) < 10.**-15
        assert abs(e[0] - e[2]) < 10.**-15
        if self._bc == BCType.FREE_SPACE:
            self._direct = kmc_direct.FreeSpaceDirect()
        elif self._bc is BCType.NEAREST:
            self._direct = kmc_direct.NearestDirect(float(e[0]))
        elif self._bc is BCType.PBC:
            self._direct = kmc_direct.PBCDirect(float(e[0]), self.domain, self.fmm.L)
        else:
            raise NotImplementedError('BCType unknown: ' + str(self._bc))


    def _get_energy(self, ids):
        """
        Get the energy of each charge in the iterable ids.

        :arg ids: Local ids of charges.
        """
        
        n = ids.shape[0]
        m = ids.shape[1]
        
        N = n * m
        ids = ids.ravel()


        h = {}
        h['old_positions']     = np.zeros((N, 3), dtype=REAL)
        h['old_fmm_cells']     = np.zeros((N, 1), dtype=INT64)
        h['old_charges']       = np.zeros((N, 1), dtype=REAL)
        h['old_energy_d']      = np.zeros((N, 1), dtype=REAL)
        h['old_energy_i']      = np.zeros((N, 1), dtype=REAL)
        h['old_ids']           = np.zeros((N, 1), dtype=INT64)
        
        h['old_positions'][:] = self.positions[ids, :].copy()
        for idi, idx in enumerate(ids):
            h['old_fmm_cells'][idi] = self.md.get_local_fmm_cell(idx)

        h['old_charges'][:] = self.charges[ids, :].copy()
        h['old_ids'][:] = self.group._kmc_fmm_order[ids, :].copy()
        
        self.kmco.get_old_energy(N, h)
        self.kmcl.get_old_energy(N, h)

        return np.sum(np.add(h['old_energy_i'], h['old_energy_d']).reshape((n, m)), axis=1)


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
    


    def _py_extract_bb_energy(self, ids):
        return self.compute_energy(self.positions[ids, :], self.charges[ids, :])
    
    def _py_extract_ab_bb_part_1(self, ids):
        assert self.comm.size == 1

        # BB interations (required to avoid double count of BB)
        BB_energy = self._py_extract_bb_energy(ids)

        # AB + BB interactions
        AB_BB_energy = 0.0
        #for ix in ids:
        #    ix = int(ix)
        #    AB_BB_energy += self._charge_indirect_energy_old(ix) + \
        #        self._direct_contrib_old(ix)
        
        if self._bc == BCType.PBC:
            tmp_field = np.zeros(len(ids), REAL)
            self._lr_energy.eval_field(
                self.positions.view[ids, :], tmp_field)

            for ixi, ix in enumerate(ids):
                tmp_field[ixi] *= self.charges.view[ix, 0]

            AB_BB_LR_energy = np.sum(tmp_field)

        else:
            AB_BB_LR_energy = 0.0
        
        return -1.0 * AB_BB_energy + BB_energy - AB_BB_LR_energy


    def _py_propose_extract(self, ids):
        
        t0 = time.time()
        assert self.comm.size == 1

        # code is written assuming the current state is A + B for A, B sets
        # of charges. B is the set to remove. Hence energy is formed of the
        # AA + AB + BB interactions.

        ids = np.array(ids, dtype=INT64)
        
        part1 = self._py_extract_ab_bb_part_1(ids)
        AB_BB_energy = 0.0
        for ix in ids:
            ix = int(ix)
            AB_BB_energy += self._charge_indirect_energy_old(ix) + \
                self._direct_contrib_old(ix)

        return (part1 - AB_BB_energy) * self.energy_unit



    def propose_extract(self, ids, use_python=False):
        """
        Propose the extraction of a set of charges by providing the local
        particle ids. Returns the change of energy if the charges were
        removed.

        :arg ids: Iterable of local charge id to remove.
        """
        
        t0 = time.time()
        assert self.comm.size == 1

        # code is written assuming the current state is A + B for A, B sets
        # of charges. B is the set to remove. Hence energy is formed of the
        # AA + AB + BB interactions.

        ids = np.array(ids, dtype=INT64)
        
        if len(ids.shape) == 1:
            return self._py_propose_extract(ids)
        elif len(ids.shape) > 2:
            raise RuntimeError('Bad ids shape')
        
        n = ids.shape[0]
        out = np.zeros(n, REAL)
        if use_python:
            for idi, idx in enumerate(ids):
                out[idi] = self._py_propose_extract(idx)
            return out


        part2 = self._get_energy(ids)
        assert len(part2) == n

        for idi in range(n):
            out[idi] = self._py_extract_ab_bb_part_1(ids[idi, :]) - part2[idi]
        
        return out * self.energy_unit



    def propose_inject(self, positions, charges):
        """
        Propose the injection of a set of charges. Returns the change in system
        energy if the set of charges were added.

        :arg positions: New charge positions.
        :arg charges: New charge values.
        """
        
        t0 = time.time()
        assert self.comm.size == 1
        N = positions.shape[0]

        BB_energy = self.compute_energy(positions, charges)

        field_values = self.eval_field(positions).reshape(N)
        AB_energy = float(np.sum(np.multiply(charges.reshape(N), field_values)))

        self._profile_inc('InjectorExtractor.propose_inject', time.time() - t0)

        return (AB_energy + BB_energy) * self.energy_unit


    def extract(self, ids):
        """
        Extract the set of charges given by local ids.

        :arg ids: Iterable of particle local ids to remove.
        """
        
        t0 = time.time()
        assert self.comm.size == 1

        with self.group.modify() as m:
            if ids is not None:
                m.remove(ids)

        self._profile_inc('InjectorExtractor.extract', time.time() - t0)
        self.initialise()


    def inject(self, add):
        """
        Inject a set of charges. e.g.

        .. highlight:: python
        .. code-block:: python

            kmc = KMCFMM(....)
            kmc.inject({
                A.P: np.array((
                    (r_x1, r_y1, r_z1),
                    (r_x2, r_y2, r_z2),
                )),
                A.Q: np.array(((-1.0), (1.0))),
                ..
            })



        :arg add: Dictonary of the style of `state.modifier.add`.
        """
        
        t0 = time.time()
        assert self.comm.size == 1
        with self.group.modify() as m:
            if add is not None:
                m.add(add)

        self.initialise()

        self._profile_inc('InjectorExtractor.inject', time.time() - t0)


class DiscoverInjectExtract:
    """
    class to identify particles which are on sites where they can be extracted
    from and empty sites where particles can be injected.

    :arg inject_sites: Tuple `((r_x, r_y, r_z),...` where particles can be injected.
    :arg extract_sites: Tuple of positions where charges can be extracted from.
    :arg positions: PositionDat to use for particle positions.
    :arg extract_flag: `ParticleDat(ncomp=1, dtype=INT64)` to use for marking potential extractions.
    """

    def __init__(self, inject_sites, extract_sites, positions, extract_flag):
        
        n_isites = len(inject_sites)
        n_esites = len(extract_sites)

        self._isites = ppmd.data.ScalarArray(ncomp=n_isites*3, dtype=REAL)
        self._iflags = ppmd.data.GlobalArray(ncomp=n_isites, dtype=INT64)

        self._esites = ppmd.data.ScalarArray(ncomp=n_esites*3, dtype=REAL)
        
        self._isites[:] = np.array(inject_sites).ravel()
        self._esites[:] = np.array(extract_sites).ravel()

        self._p = positions
        self._f = extract_flag


        kernel_src = """
        const double px = P.i[0];
        const double py = P.i[1];
        const double pz = P.i[2];

        for (int ii=0 ; ii<{N_I} ; ii++){{
            const double dx = IS[ii*3 + 0] - px;
            const double dy = IS[ii*3 + 1] - py;
            const double dz = IS[ii*3 + 2] - pz;

            const double r2 = dx*dx + dy*dy + dz*dz;
            IF[ii] += (r2 < {TOL}) ? 1 : 0;
        }}

        for (int ii=0 ; ii<{N_E} ; ii++){{
            const double dx = ES[ii*3 + 0] - px;
            const double dy = ES[ii*3 + 1] - py;
            const double dz = ES[ii*3 + 2] - pz;

            const double r2 = dx*dx + dy*dy + dz*dz;

            E.i[0] += (r2 < {TOL}) ? 1 : 0;
        }}


        """.format(
            N_I=n_isites,
            N_E=n_esites,
            TOL="0.00001"
        )


        kernel = ppmd.kernel.Kernel('inject_extract', kernel_src)
        self._loop = ppmd.loop.ParticleLoopOMP(
            kernel=kernel,
            dat_dict={
                'P': self._p(READ),
                'E': self._f(INC_ZERO),
                'IS': self._isites(READ),
                'ES': self._esites(READ),
                'IF': self._iflags(INC_ZERO)
            }
        )
    

    def __call__(self):
        """
        Mark particles that are on extract sites. Returns an array of flags
        that mark empty potential inject sites. Occupied sites are indicated
        with a value greater than 0 and empty sites a 0.
        """

        self._loop.execute()

        return self._iflags[:].copy()





















