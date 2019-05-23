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
@pytest.mark.parametrize("BC", ('free_space', '27', 'pbc'))
def test_propose_extract_1(BC):
    """
    Tests proposed moves one by one against direct calculation.
    """


    L = 12

    N = 20
    E = 2*3.1416

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()
    A.npart = N

    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)
    A.GID = data.ParticleDat(ncomp=1, dtype=INT64)

    rng = np.random.RandomState(3251)

    pi = rng.uniform(low=-0.5*E, high=0.5*E, size=(N, 3))
    qi = np.zeros((N,1), REAL)
    assert N % 2 == 0
    for px in range(N):
        qi[px, 0] = (-1)**px
    
    gi = np.arange(N).reshape((N, 1))

    with A.modify() as m:
        if MPIRANK == 0:
            m.add({
                A.P: pi[:, :],
                A.Q: qi[:, :],
                A.GID: gi[:, :]
            })


    kmc = kmc_fmm.KMCFMM(
        A.P, A.Q, A.domain, r=3, l=12, max_move=1.0, boundary_condition=BC)
    kmc.initialise()


    direct = _direct_chooser(BC, A.domain, L)
    
    # check initial energy agrees
    phi_direct_0 = direct(N, pi, qi)
    err = abs(kmc.energy - phi_direct_0) / abs(phi_direct_0)
    assert err < 10.**-5
    
    EI = kmc_inject_extract.InjectorExtractor(kmc)

    
    for testx in range(20):
        # find a +ve/-ve pair of charges

        num_remove = rng.randint(1, 10)
        remove_inds = []
        available = set(range(N))
        
        for tx in range(num_remove):
            ind = rng.randint(0, N)
            while((A.Q[ind, 0] < 0) or (ind not in available)):
                ind = rng.randint(0, N)
        
            remove_inds.append(ind)
            available.remove(ind)

        for tx in range(num_remove):
            ind = rng.randint(0, N)
            while((A.Q[ind, 0] > 0) or (ind not in available)):

                ind = rng.randint(0, N)
            remove_inds.append(ind)
            available.remove(ind)

        assert len(remove_inds) == 2*num_remove

        gids = [int(A.GID[gx, 0]) for gx in remove_inds]

        diff_extractor = EI.propose_extract(remove_inds)

        inds = set(range(N))
        for gx in gids:
            inds.remove(gx)
        inds = np.array(tuple(inds), 'int')


        # direct extract energy
        phi_direct_1 = direct(len(inds), pi[inds, :], qi[inds, :])
        diff_direct = phi_direct_1 - phi_direct_0

        err = abs(diff_extractor - diff_direct) / abs(diff_direct)
        # print(err, diff_extractor, diff_direct)

        assert err < 3.1*10.**-4


    
@pytest.mark.skipif('MPISIZE > 1')
@pytest.mark.parametrize("BC", ('free_space', '27', 'pbc'))
def test_extract_1(BC):
    """
    Tests extraction of charges, assumes propose_extract works (i.e passes above test).
    """


    L = 12

    N = 20
    E = 2*3.1416

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()
    A.npart = N

    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)
    A.GID = data.ParticleDat(ncomp=1, dtype=INT64)



    rng = np.random.RandomState(3251)

    pi = rng.uniform(low=-0.5*E, high=0.5*E, size=(N, 3))
    qi = np.zeros((N,1), REAL)
    assert N % 2 == 0
    for px in range(N):
        qi[px, 0] = (-1)**px
    
    gi = np.arange(N).reshape((N, 1))

    with A.modify() as m:
        if MPIRANK == 0:
            m.add({
                A.P: pi[:, :],
                A.Q: qi[:, :],
                A.GID: gi[:, :]
            })


    kmc = kmc_fmm.KMCFMM(
        A.P, A.Q, A.domain, r=3, l=12, max_move=1.0, boundary_condition=BC)
    kmc.initialise()


    direct = _direct_chooser(BC, A.domain, L)
    
    # check initial energy agrees
    phi_direct_0 = direct(N, pi, qi)
    err = abs(kmc.energy - phi_direct_0) / abs(phi_direct_0)
    assert err < 10.**-5
    
    EI = kmc_inject_extract.InjectorExtractor(kmc)



    
    for testx in range(70):
        # find a +ve/-ve pair of charges

        num_remove = rng.randint(1, 10)
        remove_inds = []
        available = set(range(N))
        
        for tx in range(num_remove):
            ind = rng.randint(0, N)
            while((A.Q[ind, 0] < 0) or (ind not in available)):
                ind = rng.randint(0, N)
        
            remove_inds.append(ind)
            available.remove(ind)

        for tx in range(num_remove):
            ind = rng.randint(0, N)
            while((A.Q[ind, 0] > 0) or (ind not in available)):

                ind = rng.randint(0, N)
            remove_inds.append(ind)
            available.remove(ind)

        assert len(remove_inds) == 2*num_remove


        gids = [int(A.GID[gx, 0]) for gx in remove_inds]

        diff_extractor = EI.propose_extract(remove_inds)
        EI.extract(remove_inds)

        inds = set(range(N))
        for gx in gids:
            inds.remove(gx)
        inds = np.array(tuple(inds), 'int')


        # direct extract energy
        phi_correct = phi_direct_0 + diff_extractor
        phi_test = kmc.energy
        err = abs(phi_test - phi_correct) / abs(phi_correct)


        #print(err, phi_correct, phi_test)

        assert err < 3*10.**-4

        with A.modify() as m:
            m.remove(tuple(range(A.npart_local)))
        assert A.npart == 0

        with A.modify() as m:
            if MPIRANK == 0:
                m.add({
                    A.P: pi[:, :],
                    A.Q: qi[:, :],
                    A.GID: gi[:, :]
                })

        kmc.initialise()
        err = abs(kmc.energy - phi_direct_0) / abs(phi_direct_0)
        assert err < 10.**-5











