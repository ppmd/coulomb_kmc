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

from mpi4py import MPI
MPISIZE = MPI.COMM_WORLD.Get_size()
MPIRANK = MPI.COMM_WORLD.Get_rank()
MPIBARRIER = MPI.COMM_WORLD.Barrier
DEBUG = True
SHARED_MEMORY = 'omp'

from coulomb_kmc import *

_PROFILE = common.PROFILE

REAL = ctypes.c_double
INT64 = ctypes.c_int64




@pytest.mark.parametrize("param_boundary", ('free_space', 'pbc', '27'))
def test_kmc_fmm_dat_setup_prop_mpi_1(param_boundary):
    
    L = 12
    R = 3

    N = 200
    E = 4.
    rc = E/4
    M = 8

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()
    A.npart = N
    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)
    A.prop_masks = data.ParticleDat(ncomp=M, dtype=INT64)
    A.prop_positions = data.ParticleDat(ncomp=M*3)
    A.prop_diffs = data.ParticleDat(ncomp=M)
    A.sites = data.ParticleDat(ncomp=1, dtype=INT64)

    B = state.State()
    B.domain = domain.BaseDomainHalo(extent=(E,E,E))
    B.domain.boundary_condition = domain.BoundaryTypePeriodic()
    B.npart = N
    B.P = data.PositionDat(ncomp=3)
    B.Q = data.ParticleDat(ncomp=1)

    rng  = np.random.RandomState(seed=8657)

    site_max_counts = data.ScalarArray(ncomp=8, dtype=INT64)
    site_max_counts[:] = rng.randint(0, 10, size=8)

    A.P[:] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,3))
    for px in range(N):
        A.Q[px,0] = (-1.0)**(px+1)
    bias = np.sum(A.Q[:N:, 0])/N
    A.Q[:, 0] -= bias
    A.sites[:, 0] = rng.randint(0, 8, size=N)

    
    B.P[:] = A.P.data.copy()
    B.Q[:] = A.Q.data.copy()

    A.scatter_data_from(0)
    B.scatter_data_from(0)

    # create a kmc instance
    kmc_fmmA = KMCFMM(positions=A.P, charges=A.Q, 
        domain=A.domain, r=R, l=L, boundary_condition=param_boundary)
    kmc_fmmA.initialise()
    
    kmc_fmmB = KMCFMM(positions=B.P, charges=B.Q, 
        domain=B.domain, r=R, l=L, boundary_condition=param_boundary)
    kmc_fmmB.initialise() 
    
    # print("\n arggggg")

    # for stepx in range(20):

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

        t0 = time.time()
        correct = kmc_fmmB.propose(prop)
        t1= time.time()
        to_test =  kmc_fmmA.propose_with_dats(site_max_counts, A.sites,
            A.prop_positions, A.prop_masks, A.prop_diffs, diff=True)
        t2 = time.time()


        for propi, propx in enumerate(prop):
            pid = propx[0]
            movs = propx[1]
            found_movs = 0
            for pmi in range(M):
                if A.prop_masks[pid, pmi] > 0:
                    correct_energy = correct[propi][found_movs]
                    to_test_energy = A.prop_diffs[pid, pmi] + kmc_fmmA.energy
                    
                    rel = 1.0 if abs(correct_energy) < 1 else abs(correct_energy)
                    err = abs(correct_energy - to_test_energy) / rel
                    
                    assert err < 2*(10**-15)

                    found_movs += 1
     

















