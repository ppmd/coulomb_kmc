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

c_double = ctypes.c_double

@pytest.mark.skipif("MPISIZE > 1")
def test_long_range_accept_correction_1():
    param_boundary = 'pbc'

    L = 4
    R = 3

    N = 200
    E = 4.
    rc = E/4
    M = 6

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
    prop = []
    
    nmov = 0
    # for px in range(N):
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

    correct = kmc_fmmB.md.setup_propose(prop)
    to_test =  kmc_fmmA.md.setup_propose_with_dats(site_max_counts, A.sites,
        A.prop_positions, A.prop_masks, A.prop_diffs)

    num_particles = correct[1]
    total_movs = correct[0]

    correct_arr = np.zeros((num_particles, M), dtype=c_double)
    to_test_arr = np.zeros_like(correct_arr)
    
    seed = np.random.randint(2**20)
    rng2 = np.random.RandomState(seed)

    correct_arr[:] = rng2.uniform(size=correct_arr.shape)
    to_test_arr[:] = correct_arr[:]

    # get the original corrections using python

    kmc_fmmB._si.propose(*list(list(correct) + [correct_arr]), **{'use_python': True})
    kmc_fmmA._si.propose(*list(list(to_test) + [to_test_arr]), **{'use_python': False})

    err = np.linalg.norm(correct_arr - to_test_arr, np.inf)
    assert err < 10.**-15, str(seed)


    



















