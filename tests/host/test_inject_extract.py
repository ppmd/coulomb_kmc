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

ox_range = tuple(range(-1, 2))

from kmc_test_common import *


def _direct_chooser(bc, domain, L):
    if bc == 'free_space':
        return kmc_direct.FreeSpaceDirect()
    elif bc == '27':
        return kmc_direct.NearestDirect(float(domain.extent[0]))
    elif bc == 'pbc':
        return kmc_direct.PBCDirect(float(domain.extent[0]), domain, L)


@pytest.mark.skipif('MPISIZE > 1')
def test_find_extract_inject():

    N = 100
    E = 2*3.1416

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()
    A.npart = N

    A.P = data.PositionDat(ncomp=3)
    A.GID = data.ParticleDat(ncomp=1, dtype=INT64)
    A.FLAG = data.ParticleDat(ncomp=1, dtype=INT64)

    rng = np.random.RandomState(3251)

    pi = rng.uniform(low=-0.5*E, high=0.5*E, size=(N, 3))
    gi = np.arange(N).reshape((N, 1))

    with A.modify() as m:
        if MPIRANK == 0:
            m.add({
                A.P: pi[:, :],
                A.GID: gi[:, :]
            })

    

    s = rng.permutation(N)

    id_inject_not_possible = s[:10:]
    id_extract_possible = s[10:20:]

    
    inject_sites = rng.uniform(low=-0.5*E, high=0.5*E, size=(20, 3))
    inject_sites[:10, :] = A.P.view[id_inject_not_possible, :].copy()
    extract_sites = A.P.view[id_extract_possible, :].copy()



    DIE = kmc_inject_extract.DiscoverInjectExtract(
        inject_sites, extract_sites, A.P, A.FLAG)

    
    inject_flags = DIE()

    for sx in range(10):
        # occupied sites have a 1
        assert inject_flags[sx] == 1


    for sx in range(10, 20):
        # occupied sites have a 1
        assert inject_flags[sx] == 0
    

    for px in range(N):
        if px in id_extract_possible:
            assert A.FLAG[px, 0] > 0
        else:
            assert A.FLAG[px, 0] == 0












