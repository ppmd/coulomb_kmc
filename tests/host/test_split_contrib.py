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

    All = state.State()
    All.domain = domain.BaseDomainHalo(extent=(E,E,E))
    All.domain.boundary_condition = domain.BoundaryTypePeriodic()
    All.npart = N
    All.P = data.PositionDat(ncomp=3)
    All.Q = data.ParticleDat(ncomp=1)

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
    with All.modify() as mv:
        if MPIRANK == 0:
            mv.add(
                {
                    All.P: pi,
                    All.Q: qi
                }
            )

    Near_kmc = kmc_fmm.KMCFMM(Near.P, Near.Q, Near.domain, boundary_condition='27', l=L, r=R)
    Far_kmc = kmc_fmm.KMCFMM(Far.P, Far.Q, Far.domain, boundary_condition='ff-only', l=L, r=R)
    All_kmc = kmc_fmm.KMCFMM(All.P, All.Q, All.domain, boundary_condition='pbc', l=L, r=R)

    Near_kmc.initialise()
    Far_kmc.initialise()
    All_kmc.initialise()
    
    # Check the initial energies match
    assert abs(All_kmc.energy) > 0.0
    err_energy = abs(Near_kmc.energy + Far_kmc.energy - All_kmc.energy) / abs(All_kmc.energy)
    assert err_energy < 10.**-14


    # Check eval_field is consistent
    eval_points = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,3))
    
    Near_eval = Near_kmc.eval_field(eval_points)
    Far_eval = Far_kmc.eval_field(eval_points)
    All_eval = All_kmc.eval_field(eval_points)

    err_eval_field = np.linalg.norm(Near_eval + Far_eval - All_eval, np.inf)

    assert err_eval_field < 10.**-13














