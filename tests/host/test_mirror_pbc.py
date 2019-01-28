from __future__ import print_function, division

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"

import pytest, ctypes, math
from mpi4py import MPI
import numpy as np
from itertools import product

np.set_printoptions(linewidth=200)


from ppmd import *
from ppmd.coulomb.fmm import *
from ppmd.coulomb.ewald_half import *
from scipy.special import sph_harm, lpmv
import time

from math import *

MPISIZE = MPI.COMM_WORLD.Get_size()
MPIRANK = MPI.COMM_WORLD.Get_rank()
MPIBARRIER = MPI.COMM_WORLD.Barrier
DEBUG = True
SHARED_MEMORY = 'omp'

from coulomb_kmc import *

REAL = ctypes.c_double
INT64 = ctypes.c_int64


direction_bools = (
    (True, False, False),
    (False, True, False),
    (False, False, True)
)


@pytest.mark.parametrize("direction", direction_bools)
@pytest.mark.skipif('MPISIZE > 1')
def test_kmc_fmm_pbc_1(direction):

    L = 12
    R = 3

    N = 20
    N2 = 2 * N
    E = 4.
    rc = E/4
    M = 8


    rng = np.random.RandomState(seed=8372)

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()
    A.npart = N2

    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)
    A.GID = data.ParticleDat(ncomp=1, dtype=INT64)
    A.MIRROR_MAP = data.ParticleDat(ncomp=2, dtype=INT64)
    A.prop_masks = data.ParticleDat(ncomp=M, dtype=INT64)
    A.prop_positions = data.ParticleDat(ncomp=M*3)
    A.prop_diffs = data.ParticleDat(ncomp=M)
    A.sites = data.ParticleDat(ncomp=1, dtype=INT64)

    site_max_counts = data.ScalarArray(ncomp=8, dtype=INT64)
    site_max_counts[:] = rng.randint(0, 10, size=8)

    B = state.State()
    B.domain = domain.BaseDomainHalo(extent=(E,E,E))
    B.domain.boundary_condition = domain.BoundaryTypePeriodic()
    B.npart = N2

    B.P = data.PositionDat(ncomp=3)
    B.Q = data.ParticleDat(ncomp=1)
    B.GID = data.ParticleDat(ncomp=1, dtype=INT64)
    B.MIRROR_MAP = data.ParticleDat(ncomp=2, dtype=INT64)


    half_extent = [E/2 if bx else E for bx in direction]

    S = state.State()
    S.npart = N
    S.domain = domain.BaseDomainHalo(extent=half_extent)
    S.domain.boundary_condition = domain.BoundaryTypePeriodic()
    S.P = data.PositionDat()
    S.Q = data.ParticleDat(ncomp=1)
    S.GID = data.ParticleDat(ncomp=1, dtype=ctypes.c_int64)

    for dimx in range(3):
        he = 0.25*E if direction[dimx] else 0.5*E
        S.P[:, dimx] = rng.uniform(low=-he, high=he, size=N)

    S.Q[:,0] = rng.uniform(size=N)

    S.GID[:N, 0] = np.arange(N)

    MCS = kmc_dirichlet_boundary.MirrorChargeSystem(direction, S, 'P', 'Q', 'GID')
    MS = MCS.mirror_state
    
    A.P[:N2, :] = MS.P[:N2, :]
    A.Q[:N2, :] = MS.Q[:N2, :]
    A.GID[:N2, :] = MS.GID[:N2, :]
    A.MIRROR_MAP[:N2, :]  = MS.mirror_map[:N2, :]
    A.sites[:, 0] = rng.randint(0, 8, size=N2)

    B.P[:N2, :] = MS.P[:N2, :]
    B.Q[:N2, :] = MS.Q[:N2, :]
    B.GID[:N2, :] = MS.GID[:N2, :]
    B.MIRROR_MAP[:N2, :]  = MS.mirror_map[:N2, :]

    A.scatter_data_from(0)
    B.scatter_data_from(0)

    fmm_bc = False
    kmc_bc = 'pbc'

    kmc_fmmA = KMCFMM(positions=A.P, charges=A.Q, domain=A.domain, r=R, l=L, 
        boundary_condition=kmc_bc, mirror_direction=direction)
    kmc_fmmA.initialise()
    
    fmm = PyFMM(B.domain, N=N2, free_space=fmm_bc, r=kmc_fmmA.fmm.R, l=kmc_fmmA.fmm.L)

    def _mirror_pos(rpos):
        f = [-1.0 if dx else 1.0 for dx in direction]
        return (rpos[0] * f[0], rpos[1] * f[1], rpos[2] * f[2])

    for testx in range(4):
        prop = []
        nmov = 0
        for px in range(N):
            tmp = []
            masks = np.zeros(M)
            masks[:site_max_counts[A.sites[px,0]]:] = 1
            masks = rng.permutation(masks)

            for propx in range(M):
                mask = masks[propx]
                prop_pos = rng.uniform(low=-0.5*E, high=0.5*E, size=3)
                A.prop_masks[px, propx] = mask
                A.prop_positions[px, propx*3:propx*3+3:] = prop_pos
                
                if mask > 0:
                    tmp.append(list(prop_pos))
                    nmov += 1
            if len(tmp) > 0:
                prop.append((px, np.array(tmp)))

        to_test_2 = kmc_fmmA.propose(prop)
        to_test =  kmc_fmmA.propose_with_dats(site_max_counts, A.sites,
            A.prop_positions, A.prop_masks, A.prop_diffs, diff=True)
        t2 = time.time()
        
        # test mirror charge proposals against a full fmm solve
        for propi, propx in enumerate(prop):
            pid = propx[0]
            movs = propx[1]
            for mxi in range(propx[1].shape[0]):

                mid = B.MIRROR_MAP[pid, 0]
                mgid = np.where(B.GID[:] == mid)[0][0]               

                pos = movs[mxi, :]
                mpos = np.array(_mirror_pos(pos))

                old_pos = B.P[pid, :]
                old_mpos = B.P[mgid, :]
                B.P[pid, :] = pos
                B.P[mgid, :] = mpos
                correct = fmm(B.P, B.Q)
                B.P[pid, :] = old_pos
                B.P[mgid, :] = old_mpos

                kmc_e = to_test_2[propi][mxi]
                err = abs(correct - kmc_e) / abs(correct)
                assert err < 5*(10.**-5)


        # test the python propose interface against the dat interface
        for propi, propx in enumerate(prop):
            pid = propx[0]
            movs = propx[1]
            found_movs = 0
            for pmi in range(M):
                if A.prop_masks[pid, pmi] > 0:
                    
                    t2 = to_test_2[propi][found_movs]

                    to_test_energy = A.prop_diffs[pid, pmi] + kmc_fmmA.energy
                    
                    rel = 1.0 if abs(t2) < 1 else abs(t2)
                    err = abs(t2 - to_test_energy) / rel
                    
                    assert err < 2*(10**-14)

                    found_movs += 1


 

@pytest.mark.parametrize("direction", direction_bools)
@pytest.mark.skipif('MPISIZE > 1')
def test_kmc_fmm_pbc_2(direction):

    L = 12
    R = 3

    N = 20
    N2 = 2 * N
    E = 4.
    rc = E/4
    M = 8


    rng = np.random.RandomState(seed=8372)

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()
    A.npart = N2

    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)
    A.GID = data.ParticleDat(ncomp=1, dtype=INT64)
    A.MIRROR_MAP = data.ParticleDat(ncomp=2, dtype=INT64)
    A.prop_masks = data.ParticleDat(ncomp=M, dtype=INT64)
    A.prop_positions = data.ParticleDat(ncomp=M*3)
    A.prop_diffs = data.ParticleDat(ncomp=M)
    A.sites = data.ParticleDat(ncomp=1, dtype=INT64)

    site_max_counts = data.ScalarArray(ncomp=8, dtype=INT64)
    site_max_counts[:] = rng.randint(0, 10, size=8)

    B = state.State()
    B.domain = domain.BaseDomainHalo(extent=(E,E,E))
    B.domain.boundary_condition = domain.BoundaryTypePeriodic()
    B.npart = N2

    B.P = data.PositionDat(ncomp=3)
    B.Q = data.ParticleDat(ncomp=1)
    B.GID = data.ParticleDat(ncomp=1, dtype=INT64)
    B.MIRROR_MAP = data.ParticleDat(ncomp=2, dtype=INT64)


    half_extent = [E/2 if bx else E for bx in direction]

    S = state.State()
    S.npart = N
    S.domain = domain.BaseDomainHalo(extent=half_extent)
    S.domain.boundary_condition = domain.BoundaryTypePeriodic()
    S.P = data.PositionDat()
    S.Q = data.ParticleDat(ncomp=1)
    S.GID = data.ParticleDat(ncomp=1, dtype=ctypes.c_int64)

    for dimx in range(3):
        he = 0.25*E if direction[dimx] else 0.5*E
        S.P[:, dimx] = rng.uniform(low=-he, high=he, size=N)

    S.Q[:,0] = rng.uniform(size=N)

    S.GID[:N, 0] = np.arange(N)

    MCS = kmc_dirichlet_boundary.MirrorChargeSystem(direction, S, 'P', 'Q', 'GID')
    MS = MCS.mirror_state
    
    A.P[:N2, :] = MS.P[:N2, :]
    A.Q[:N2, :] = MS.Q[:N2, :]
    A.GID[:N2, :] = MS.GID[:N2, :]
    A.MIRROR_MAP[:N2, :]  = MS.mirror_map[:N2, :]
    A.sites[:, 0] = rng.randint(0, 8, size=N2)

    B.P[:N2, :] = MS.P[:N2, :]
    B.Q[:N2, :] = MS.Q[:N2, :]
    B.GID[:N2, :] = MS.GID[:N2, :]
    B.MIRROR_MAP[:N2, :]  = MS.mirror_map[:N2, :]

    A.scatter_data_from(0)
    B.scatter_data_from(0)

    fmm_bc = False
    kmc_bc = 'pbc'

    kmc_fmmA = KMCFMM(positions=A.P, charges=A.Q, domain=A.domain, r=R, l=L, 
        boundary_condition=kmc_bc, mirror_direction=direction)
    kmc_fmmA.initialise()
    
    fmm = PyFMM(B.domain, N=N2, free_space=fmm_bc, r=kmc_fmmA.fmm.R, l=kmc_fmmA.fmm.L)
    fmmA = PyFMM(A.domain, N=N2, free_space=fmm_bc, r=kmc_fmmA.fmm.R, l=kmc_fmmA.fmm.L)

    def _make_prop_pos():
        p = [0,0,0]
        for dimx in range(3):
            he = 0 if direction[dimx] else 0.5*E
            p[dimx] = rng.uniform(low=-0.5*E, high=he, size=1)[0]
        return p

    def _mirror_pos(rpos):
        f = [-1.0 if dx else 1.0 for dx in direction]
        return (rpos[0] * f[0], rpos[1] * f[1], rpos[2] * f[2])


    for testx in range(20):
        pid = rng.randint(0, N2)

        mid = B.MIRROR_MAP[pid, 0]
        mgid = np.where(B.GID[:] == mid)[0][0]               

        pos = _make_prop_pos()
        mpos = np.array(_mirror_pos(pos))

        old_pos = B.P[pid, :]
        old_mpos = B.P[mgid, :]

        B.P[pid, :] = pos
        B.P[mgid, :] = mpos

        correct = fmm(B.P, B.Q)
        
        kmc_fmmA.accept(
            (
                (pid, pos),
                (mgid, mpos)
            )
        )

        to_test = kmc_fmmA.energy

        err = abs(correct - to_test) / abs(correct)
        assert err < 2*(10.**-5)

        to_test = fmmA(A.P, A.Q)
        err = abs(correct - to_test) / abs(correct)
        assert err < 2*(10.**-5)




