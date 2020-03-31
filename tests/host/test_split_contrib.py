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

REAL = ctypes.c_double
INT64 = ctypes.c_int64


def test_split_1():
    L = 12
    R = 3

    N = 200
    E = 1.
    rc = E/4

    rng = np.random.RandomState(seed=12415)

    Near = state.State()
    Near.domain = domain.BaseDomainHalo(extent=(E,E,E))
    Near.domain.boundary_condition = domain.BoundaryTypePeriodic()
    Near.npart = N
    Near.P = data.PositionDat(ncomp=3)
    Near.Q = data.ParticleDat(ncomp=1)

    Far = state.State()
    Far.domain = domain.BaseDomainHalo(extent=(E,E,E))
    Far.domain.boundary_condition = domain.BoundaryTypePeriodic()
    Far.npart = N
    Far.P = data.PositionDat(ncomp=3)
    Far.Q = data.ParticleDat(ncomp=1)

    pi = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,3))
    qi = rng.uniform(size=(N, 1))
    qi -= np.sum(qi) / N

    with Far.modify() as mv:
        if MPIRANK == 0:
            mv.add(
                {
                    Far.P: pi,
                    Far.Q: qi
                }
            )
    with Near.modify() as mv:
        if MPIRANK == 0:
            mv.add(
                {
                    Near.P: pi,
                    Near.Q: qi
                }
            )
    
    Near_kmc = kmc_fmm.KMCFMM(Near.P, Near.Q, Near.domain, boundary_condition='27', l=L, r=R)
    Far_kmc = kmc_fmm.KMCFMM(Far.P, Far.Q, Far.domain, boundary_condition='ff-only', l=L, r=R)










