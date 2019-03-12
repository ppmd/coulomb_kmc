from __future__ import print_function, division



__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"

import pytest, ctypes, math
import numpy as np
from itertools import product

np.set_printoptions(linewidth=200)

from ppmd import *
from ppmd.coulomb.fmm import *
from ppmd.coulomb.ewald_half import *
from scipy.special import sph_harm, lpmv
import time

from math import *

# from mpi4py import MPI
# MPISIZE = MPI.COMM_WORLD.Get_size()
# MPIRANK = MPI.COMM_WORLD.Get_rank()
# MPIBARRIER = MPI.COMM_WORLD.Barrier
# DEBUG = True
# SHARED_MEMORY = 'omp'

from coulomb_kmc import *

_PROFILE = common.PROFILE

REAL = ctypes.c_double
INT64 = ctypes.c_int64


MPI = mpi.MPI




@pytest.mark.parametrize("param_boundary", ('free_space', 'pbc', '27'))
def test_kmc_multilevel_1(param_boundary):
    
    L = 16

    N = 1000
    E = 4.
    rc = E/4
    M = 8

    
    ABC = [state.State(), state.State(), state.State()]
    kmc_fmm = []
    rset = [3,4,5]
    for Ai, A in enumerate(ABC):
        A.domain = domain.BaseDomainHalo(extent=(E,E,E))


        A.domain.boundary_condition = domain.BoundaryTypePeriodic()
        A.npart = N
        A.P = data.PositionDat(ncomp=3)
        A.Q = data.ParticleDat(ncomp=1)
        A.prop_masks = data.ParticleDat(ncomp=M, dtype=INT64)
        A.prop_positions = data.ParticleDat(ncomp=M*3)
        A.prop_diffs = data.ParticleDat(ncomp=M)
        A.sites = data.ParticleDat(ncomp=1, dtype=INT64)

        kmc_fmm.append(KMCFMM(positions=A.P, charges=A.Q, domain=A.domain, r=rset[Ai], l=L, boundary_condition=param_boundary))

    rng  = np.random.RandomState(seed=8657)

    A = ABC[0]

    site_max_counts = data.ScalarArray(ncomp=8, dtype=INT64)
    site_max_counts[:] = rng.randint(0, 10, size=8)

    A.P[:] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,3))
    for px in range(N):
        A.Q[px,0] = (-1.0)**(px+1)
    bias = np.sum(A.Q[:N:, 0])/N
    A.Q[:, 0] -= bias
    A.sites[:, 0] = rng.randint(0, 8, size=N)

    for B in ABC[1:]:
        B.P[:] = A.P[:]
        B.Q[:] = A.Q[:]
        B.sites[:] = A.sites[:]
    
    
    for B in ABC:
        B.scatter_data_from(0)

    
    for k in kmc_fmm:
        k.initialise()

    ce = kmc_fmm[0].energy
    for k in kmc_fmm[1:]:
        kce = k.energy
        rel = abs(ce) if abs(ce) > 0 else 1.0
        assert abs(ce - kce) / rel < 10.**-4


    for testx in range(4):
        prop = []
        nmov = 0
        for px in range(A.npart_local):
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
        

        for B in ABC[1:]:
            assert A.npart_local == B.npart_local
            n = A.npart_local
            assert np.linalg.norm(A.P[:n,:].ravel() - B.P[:n,:].ravel(), np.inf) < 10.**-14
            assert np.linalg.norm(A.Q[:n,:].ravel() - B.Q[:n,:].ravel(), np.inf) < 10.**-14
            B.prop_masks[:n, :] = A.prop_masks[:n,:]
            B.prop_positions[:n, :] = A.prop_positions[:n, :]


        for Bi, B in enumerate(ABC):
            kmc_fmm[Bi].propose_with_dats(site_max_counts, B.sites, B.prop_positions, B.prop_masks, B.prop_diffs, diff=True)

        for B in ABC[1:]:
            for propi, propx in enumerate(prop):
                pid = propx[0]
                movs = propx[1]
                for pmi in range(M):
                    if A.prop_masks[pid, pmi] > 0:
                        energy_3 = A.prop_diffs[pid, pmi]
                        energy_r = B.prop_diffs[pid, pmi]
                        
                        rel = abs(energy_3) if abs(energy_3) > 0 else 1.0
                        
                        # this diff is tiny 
                        if rel < 0.1:
                            assert abs(energy_3 - energy_r) < 10.**-4
                        else:
                            assert abs(energy_3 - energy_r) / rel < 2*10.**-4


    for k in kmc_fmm:
        k.free()












