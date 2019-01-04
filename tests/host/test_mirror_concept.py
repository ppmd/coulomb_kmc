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



@pytest.mark.skipif('MPISIZE > 1')
def test_kmc_fmm_eval_field_1():
    """
    Tests proposed moves one by one against direct calculation.
    Considers the primary image and the nearest 27 neighbours.
    """

    L = 12
    R = 3

    N = 50
    N2 = 10
    E = 4.
    rc = E/4

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()
    A.npart = N

    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)



    rng = np.random.RandomState(seed=1234)

    A.P[:] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,3))
    for px in range(N):
        A.Q[px,0] = (-1.0)**(px+1)
    bias = np.sum(A.Q[:N:, 0])/N
    A.Q[:, 0] -= bias
    
    A.scatter_data_from(0)
    
    bcs = 'free_space'
 
    kmc_fmm = KMCFMM(positions=A.P, charges=A.Q, 
        domain=A.domain, r=R, l=L, boundary_condition=bcs)
    kmc_fmm.initialise()


    eval_points = rng.uniform(low=-0.5*E, high=0.5*E, size=(N2, 3))

    correct_field = np.zeros(N2, dtype=REAL)
    for fx in range(N2):
        tmp = 0.0
        ptmp = eval_points[fx, :]
        for px in range(N):
            q = A.Q[px, 0]
            tmp += q / np.linalg.norm(ptmp - A.P[px, :])
        correct_field[fx] = tmp
    
    
    kmc_field = kmc_fmm.eval_field(eval_points)

    err = np.linalg.norm(correct_field - kmc_field, np.inf)
    assert err < 10.**-5








