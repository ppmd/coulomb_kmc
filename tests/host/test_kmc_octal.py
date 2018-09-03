from __future__ import print_function, division

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"

import pytest, ctypes, math
from mpi4py import MPI
import numpy as np

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


def test_kmc_octal_1():
    eps = 10.**-5
    L = 12
    R = 3

    N = 20
    E = 4.
    rc = E/4

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()
    A.npart = N

    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)
    A.PP = data.ParticleDat(ncomp=3)

    A.crr = data.ScalarArray(ncomp=1)

    rng = np.random.RandomState(seed=8657)

    if N == 4:
        ra = 0.25 * E
        nra = -0.25 * E

        A.P[0,:] = ( 1.6,  1.6, 0.0)
        A.P[1,:] = (-1.500001,  1.499999, 0.0)
        A.P[2,:] = (-1.500001, -1.500001, 0.0)
        A.P[3,:] = ( 0.0,  0.0, 0.0)

        A.Q[0,0] = -1.
        A.Q[1,0] = 1.
        A.Q[2,0] = -1.
        A.Q[3,0] = 0.
    else:
        A.P[:] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,3))
        for px in range(N):
            A.Q[px,0] = (-1.0)**(px+1)
        bias = np.sum(A.Q[:N:, 0])/N
        A.Q[:, 0] -= bias

    A.scatter_data_from(0)
    
    # create a kmc instance
    kmc_fmm = KMCFMM(positions=A.P, charges=A.Q, 
        domain=A.domain, r=R, l=L, boundary_condition='free_space')
    kmc_fmm.initialise()
    
    print("\n")
    kmco = kmc_octal.LocalCellExpansions(kmc_fmm.fmm, 1.0)
    kmco.initialise()





