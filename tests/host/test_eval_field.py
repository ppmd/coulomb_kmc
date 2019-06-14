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
@pytest.mark.parametrize("R", (4, 5))
@pytest.mark.skipif("MPI.COMM_WORLD.size > 1")
def test_direct_eval_field(param_boundary, R):
    
    L = 12

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

    kmc_fmm = KMCFMM(positions=A.P, charges=A.Q, domain=A.domain, r=R, l=L, boundary_condition=param_boundary)

    rng  = np.random.RandomState(seed=8657)


    pi = rng.uniform(low=-0.5*E, high=0.5*E, size=(N, 3))
    qi = np.zeros((N, 1), REAL)
    for px in range(N):
        qi[px,0] = (-1.0)**(px+1)
    bias = np.sum(qi[:N:, 0])/N
    qi[:, 0] -= bias

    with A.modify() as m:
        m.add({
            A.P: pi,
            A.Q: qi
        })


    kmc_fmm.initialise()

    
    m1 = 100
    m2 = 10
    for testx in range(m1):
        pos = rng.uniform(-0.5*E, 0.5*E, (m2, 3))

        correct = np.zeros(m2, REAL)
        to_test = np.zeros(m2, REAL)
        
        for pointx in range(m2):
            correct[pointx] = kmc_fmm._direct_contrib_new(None, pos[pointx,:])
        
        kmc_fmm.kmcl.eval_field(pos, to_test)

        err = np.linalg.norm(correct - to_test, np.inf)

        assert err < 10.**-14


    kmc_fmm.free()












@pytest.mark.parametrize("param_boundary", ('free_space', 'pbc', '27'))
@pytest.mark.parametrize("R", (4, 5))
@pytest.mark.skipif("MPI.COMM_WORLD.size > 1")
def test_indirect_eval_field(param_boundary, R):
    
    L = 12

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

    kmc_fmm = KMCFMM(positions=A.P, charges=A.Q, domain=A.domain, r=R, l=L, boundary_condition=param_boundary)

    rng  = np.random.RandomState(seed=8657)


    pi = rng.uniform(low=-0.5*E, high=0.5*E, size=(N, 3))
    qi = np.zeros((N, 1), REAL)
    for px in range(N):
        qi[px,0] = (-1.0)**(px+1)
    bias = np.sum(qi[:N:, 0])/N
    qi[:, 0] -= bias

    with A.modify() as m:
        m.add({
            A.P: pi,
            A.Q: qi
        })


    kmc_fmm.initialise()

    
    m1 = 100
    m2 = 10
    for testx in range(m1):
        pos = rng.uniform(-0.5*E, 0.5*E, (m2, 3))

        correct = np.zeros(m2, REAL)
        to_test = np.zeros(m2, REAL)
        
        for pointx in range(m2):
            correct[pointx] = kmc_fmm._charge_indirect_energy_new(None, pos[pointx,:])
        
        kmc_fmm.kmco.eval_field(pos, to_test)

        err = np.linalg.norm(correct - to_test, np.inf)

        #print(err, correct, to_test)

        assert err < 10.**-14


    kmc_fmm.free()
