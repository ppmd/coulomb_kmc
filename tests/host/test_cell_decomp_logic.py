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
from ppmd.coulomb.direct import *

_PROFILE = common.PROFILE

REAL = ctypes.c_double
INT64 = ctypes.c_int64


MPI = mpi.MPI


def test_array_lcell_to_gcell():
    
    param_boundary = 'pbc'
    L = 4
    R = 5
    N = 200
    E = 4.0
    rc = E / 4
    M = 8

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E, E, E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()
    A.npart = N
    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)

    kmc_fmm = KMCFMM(positions=A.P, charges=A.Q, domain=A.domain, r=R, l=L, boundary_condition=param_boundary, max_move=0.1)

    rng = np.random.RandomState(seed=8657)

    pi = rng.uniform(low=-0.5 * E, high=0.5 * E, size=(N, 3))
    qi = np.zeros((N, 1), REAL)
    for px in range(N):
        qi[px, 0] = (-1.0) ** (px + 1)
    bias = np.sum(qi[:N:, 0]) / N
    qi[:, 0] -= bias

    with A.modify() as m:
        if MPI.COMM_WORLD.rank == 0:
            m.add({A.P: pi, A.Q: qi})

    kmc_fmm.initialise()

    ids = np.array(range(A.npart_local), INT64)

    to_test_array = kmc_fmm.md.get_local_fmm_cell_array(ids)
    for ix in range(A.npart_local):
        correct = kmc_fmm.md.get_local_fmm_cell(ix)
        to_test = to_test_array[ix]

        assert correct == to_test

    kmc_fmm.free()





