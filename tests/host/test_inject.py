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
def test_propose_inject_1(BC):
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

    pi = np.array(rng.uniform(low=-0.5*E, high=0.5*E, size=(N, 3)), REAL)
    qi = np.zeros((N,1), REAL)
    assert N % 2 == 0
    for px in range(N):
        qi[px, 0] = (-1)**px
    
    gi = np.arange(N).reshape((N, 1))

    kmc = kmc_fmm.KMCFMM(
        A.P, A.Q, A.domain, r=3, l=12, max_move=1.0, boundary_condition=BC)

    direct = _direct_chooser(BC, A.domain, L)



    phi_direct = direct(N, pi[:, :], qi[:, :])


    for testx in range(20):


        num_add = rng.randint(1, 10)
        add_inds = []
        available = set(range(N))
        
        for tx in range(num_add):
            ind = rng.randint(0, N)
            while((qi[ind, 0] < 0) or (ind not in available)):
                ind = rng.randint(0, N)
        
            add_inds.append(ind)
            available.remove(ind)

        for tx in range(num_add):
            ind = rng.randint(0, N)
            while((qi[ind, 0] > 0) or (ind not in available)):

                ind = rng.randint(0, N)
            add_inds.append(ind)
            available.remove(ind)

        assert len(add_inds) == 2*num_add


        gids = [int(gi[gx, 0]) for gx in add_inds]

        inds = set(range(N))
        for gx in gids:
            inds.remove(gx)
        inds = np.array(tuple(inds), 'int')


        with A.modify() as m:
            if MPIRANK == 0:
                m.add({
                    A.P: pi[inds, :],
                    A.Q: qi[inds, :],
                    A.GID: gi[inds, :]
                })


        kmc.initialise()

        # check initial energy agrees
        phi_direct_0 = direct(A.npart_local, A.P.view, A.Q.view)

        err = abs(kmc.energy - phi_direct_0) / abs(phi_direct_0)
        assert err < 10.**-5
            
        
        diff_injector = kmc.propose_inject(
            pi[np.array(gids), :],
            qi[np.array(gids), :],
        )


        # direct extract energy
        diff_direct = phi_direct - phi_direct_0

        err = abs(diff_injector - diff_direct) / abs(diff_direct)

        assert err < 2*(10**-4)

        # print(err, diff_injector, diff_direct)

        

        with A.modify() as m:
            m.remove(tuple(range(A.npart_local)))
        assert A.npart == 0







@pytest.mark.skipif('MPISIZE > 1')
@pytest.mark.parametrize("BC", ('free_space', '27', 'pbc'))
def test_inject_1(BC):


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

    pi = np.array(rng.uniform(low=-0.5*E, high=0.5*E, size=(N, 3)), REAL)
    qi = np.zeros((N,1), REAL)
    assert N % 2 == 0
    for px in range(N):
        qi[px, 0] = (-1)**px
    
    gi = np.arange(N).reshape((N, 1))

    kmc = kmc_fmm.KMCFMM(
        A.P, A.Q, A.domain, r=3, l=12, max_move=1.0, boundary_condition=BC)

    direct = _direct_chooser(BC, A.domain, L)



    phi_direct = direct(N, pi[:, :], qi[:, :])


    for testx in range(20):


        num_add = rng.randint(1, 10)
        add_inds = []
        available = set(range(N))
        
        for tx in range(num_add):
            ind = rng.randint(0, N)
            while((qi[ind, 0] < 0) or (ind not in available)):
                ind = rng.randint(0, N)
        
            add_inds.append(ind)
            available.remove(ind)

        for tx in range(num_add):
            ind = rng.randint(0, N)
            while((qi[ind, 0] > 0) or (ind not in available)):

                ind = rng.randint(0, N)
            add_inds.append(ind)
            available.remove(ind)

        assert len(add_inds) == 2*num_add


        gids = [int(gi[gx, 0]) for gx in add_inds]

        inds = set(range(N))
        for gx in gids:
            inds.remove(gx)
        inds = np.array(tuple(inds), 'int')


        with A.modify() as m:
            if MPIRANK == 0:
                m.add({
                    A.P: pi[inds, :],
                    A.Q: qi[inds, :],
                    A.GID: gi[inds, :]
                })


        kmc.initialise()
        

        i = np.array(add_inds, 'int')
        kmc.inject({
            A.P: pi[i,:],
            A.Q: qi[i,:],
            A.GID: gi[i,:]
        })

        phi = kmc.energy
        err = abs(phi - phi_direct) / abs(phi_direct)
        
        assert err < 10.**-5
        

        with A.modify() as m:
            m.remove(tuple(range(A.npart_local)))
        assert A.npart == 0






@pytest.mark.skipif('MPISIZE > 1')
@pytest.mark.parametrize("BC", ('pbc', ))
def test_propose_inject_split_1(BC):
    """
    Tests proposed moves one by one against direct calculation.
    """


    L = 16

    N = 20
    E = 2*3.1416

    NEAR = state.State(
        domain=domain.BaseDomainHalo(
            extent=(E,E,E),
            boundary_condition=domain.BoundaryTypePeriodic()
        ),
        particle_dats={
            'P': data.PositionDat(ncomp=3),
            'Q': data.ParticleDat(ncomp=1),
            'GID': data.ParticleDat(ncomp=1, dtype=INT64),
        }
    )

    FAR = state.State(
        domain=domain.BaseDomainHalo(
            extent=(E,E,E),
            boundary_condition=domain.BoundaryTypePeriodic()
        ),
        particle_dats={
            'P': data.PositionDat(ncomp=3),
            'Q': data.ParticleDat(ncomp=1),
            'GID': data.ParticleDat(ncomp=1, dtype=INT64),
        }
    )

    rng = np.random.RandomState(3251)

    pi = np.array(rng.uniform(low=-0.5*E, high=0.5*E, size=(N, 3)), REAL)
    qi = np.zeros((N,1), REAL)
    assert N % 2 == 0
    for px in range(N):
        qi[px, 0] = (-1)**px
    
    gi = np.arange(N).reshape((N, 1))

    FAR_KMC = kmc_fmm.KMCFMM(FAR.P, FAR.Q, FAR.domain, r=3, l=L, max_move=1.0, boundary_condition='ff-only')
    NEAR_KMC = kmc_fmm.KMCFMM(NEAR.P, NEAR.Q, NEAR.domain, r=3, l=L, max_move=1.0, boundary_condition='27')

    direct = _direct_chooser(BC, FAR_KMC.domain, L)
    phi_direct = direct(N, pi[:, :], qi[:, :])


    for testx in range(20):


        num_add = rng.randint(1, 10)
        add_inds = []
        available = set(range(N))
        
        for tx in range(num_add):
            ind = rng.randint(0, N)
            while((qi[ind, 0] < 0) or (ind not in available)):
                ind = rng.randint(0, N)
        
            add_inds.append(ind)
            available.remove(ind)

        for tx in range(num_add):
            ind = rng.randint(0, N)
            while((qi[ind, 0] > 0) or (ind not in available)):

                ind = rng.randint(0, N)
            add_inds.append(ind)
            available.remove(ind)

        assert len(add_inds) == 2*num_add


        gids = [int(gi[gx, 0]) for gx in add_inds]

        inds = set(range(N))
        for gx in gids:
            inds.remove(gx)
        inds = np.array(tuple(inds), 'int')


        for sx in (FAR, NEAR):
            with sx.modify() as m:
                if MPIRANK == 0:
                    m.add({
                        sx.P: pi[inds, :],
                        sx.Q: qi[inds, :],
                        sx.GID: gi[inds, :]
                    })


        FAR_KMC.initialise()
        NEAR_KMC.initialise()

        # check initial energy agrees
        phi_direct_0 = direct(FAR.npart_local, FAR.P.view, FAR.Q.view)
        diff_move = (pi[np.array(gids), :], qi[np.array(gids), :])
        # direct extract energy
        diff_direct = phi_direct - phi_direct_0

    
        NEAR_diff = NEAR_KMC.propose_inject(*diff_move)
        FAR_diff = FAR_KMC.propose_inject(*diff_move)

        to_test = NEAR_diff + FAR_diff

        err = abs(to_test - diff_direct) / abs(diff_direct)

        assert err < 10.**-5


        for sx in (FAR, NEAR):
            with sx.modify() as m:
                m.remove(tuple(range(sx.npart_local)))
            assert sx.npart == 0

@pytest.mark.skipif('MPISIZE > 1')
@pytest.mark.parametrize("BC", ('pbc', ))
def test_inject_split_1(BC):


    L = 12

    N = 20
    E = 2*3.1416



    NEAR = state.State(
        domain=domain.BaseDomainHalo(
            extent=(E,E,E),
            boundary_condition=domain.BoundaryTypePeriodic()
        ),
        particle_dats={
            'P': data.PositionDat(ncomp=3),
            'Q': data.ParticleDat(ncomp=1),
            'GID': data.ParticleDat(ncomp=1, dtype=INT64),
        }
    )

    FAR = state.State(
        domain=domain.BaseDomainHalo(
            extent=(E,E,E),
            boundary_condition=domain.BoundaryTypePeriodic()
        ),
        particle_dats={
            'P': data.PositionDat(ncomp=3),
            'Q': data.ParticleDat(ncomp=1),
            'GID': data.ParticleDat(ncomp=1, dtype=INT64),
        }
    )



    rng = np.random.RandomState(3251)

    pi = np.array(rng.uniform(low=-0.5*E, high=0.5*E, size=(N, 3)), REAL)
    qi = np.zeros((N,1), REAL)
    assert N % 2 == 0
    for px in range(N):
        qi[px, 0] = (-1)**px
    
    gi = np.arange(N).reshape((N, 1))


    FAR_KMC = kmc_fmm.KMCFMM(FAR.P, FAR.Q, FAR.domain, r=3, l=L, max_move=1.0, boundary_condition='ff-only')
    NEAR_KMC = kmc_fmm.KMCFMM(NEAR.P, NEAR.Q, NEAR.domain, r=3, l=L, max_move=1.0, boundary_condition='27')

    FAR_KMC.initialise()
    NEAR_KMC.initialise()

    direct = _direct_chooser(BC, FAR.domain, L)
    phi_direct = direct(N, pi[:, :], qi[:, :])


    for testx in range(20):


        num_add = rng.randint(1, 10)
        add_inds = []
        available = set(range(N))
        
        for tx in range(num_add):
            ind = rng.randint(0, N)
            while((qi[ind, 0] < 0) or (ind not in available)):
                ind = rng.randint(0, N)
        
            add_inds.append(ind)
            available.remove(ind)

        for tx in range(num_add):
            ind = rng.randint(0, N)
            while((qi[ind, 0] > 0) or (ind not in available)):

                ind = rng.randint(0, N)
            add_inds.append(ind)
            available.remove(ind)

        assert len(add_inds) == 2*num_add


        gids = [int(gi[gx, 0]) for gx in add_inds]

        inds = set(range(N))
        for gx in gids:
            inds.remove(gx)
        inds = np.array(tuple(inds), 'int')

        for sx in (FAR, NEAR):
            with sx.modify() as m:
                if MPIRANK == 0:
                    m.add({
                        sx.P: pi[inds, :],
                        sx.Q: qi[inds, :],
                        sx.GID: gi[inds, :]
                    })


        FAR_KMC.initialise()
        NEAR_KMC.initialise()

        i = np.array(add_inds, 'int')

        FAR_KMC.inject({
            FAR.P: pi[i,:],
            FAR.Q: qi[i,:],
            FAR.GID: gi[i,:]
        })

        NEAR_KMC.inject({
            NEAR.P: pi[i,:],
            NEAR.Q: qi[i,:],
            NEAR.GID: gi[i,:]
        })


        to_test = FAR_KMC.energy + NEAR_KMC.energy

        err = abs(to_test - phi_direct) / abs(phi_direct)
        assert err < 10.**-5

        for sx in (FAR, NEAR):
            with sx.modify() as m:
                m.remove(tuple(range(sx.npart_local)))
            assert sx.npart == 0






