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


@pytest.mark.skipif('MPISIZE > 1')
def test_kmc_fmm_nearest_27_1():
    """
    Tests proposed moves one by one against direct calculation.
    Considers the primary image and the nearest 27 neighbours.
    """

    eps = 10.**-5
    L = 12
    R = 3

    N = 50
    E = 4.
    rc = E/4

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()
    A.npart = N

    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)


    B = state.State()
    B.domain = domain.BaseDomainHalo(extent=(E,E,E))
    B.domain.boundary_condition = domain.BoundaryTypePeriodic()
    B.npart = N

    B.P = data.PositionDat(ncomp=3)
    B.Q = data.ParticleDat(ncomp=1)


    rng = np.random.RandomState(seed=1234)

    if N == 4:
        ra = 0.25 * E
        nra = -0.25 * E

        A.P[0,:] = ( 0.24,  0.24, 0.0)
        A.P[1,:] = (-0.24,  0.24, 0.0)
        A.P[2,:] = (-0.24, -0.24, 0.0)
        A.P[3,:] = ( 0.24, -0.24, 0.0)

        A.Q[0,0] = 1.
        A.Q[1,0] = 1.
        A.Q[2,0] = 1.
        A.Q[3,0] = 1.
    
    elif N == 1:

        A.P[0,:] = ( -1.99, -1.99, -1.99)
        A.Q[0,0] = -1.


    else:
        A.P[:] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,3))
        for px in range(N):
            A.Q[px,0] = (-1.0)**(px+1)
        bias = np.sum(A.Q[:N:, 0])/N
        A.Q[:, 0] -= bias
    
    B.P[:] = A.P[:]
    B.Q[:] = A.Q[:]

    A.scatter_data_from(0)
    B.scatter_data_from(0)
    
 
    kmc_fmm = KMCFMM(positions=A.P, charges=A.Q, 
        domain=A.domain, r=R, l=L, boundary_condition='27')
    kmc_fmm.initialise()

    fmm = PyFMM(B.domain, N=N, free_space='27', r=kmc_fmm.fmm.R, l=kmc_fmm.fmm.L)

    def _direct():
        _phi_direct = fmm(positions=B.P, charges=B.Q)
        return _phi_direct
    
    phi_direct = _direct()


    for rx in range(N):
        pid = rng.randint(0, N-1)
        pos = rng.uniform(low=-0.5*E, high=0.5*E, size=3)

        old_pos = B.P[pid, :]
        B.P[pid, :] = pos
        phi_direct = _direct()
        B.P[pid, :] = old_pos 

        prop_energy = kmc_fmm.test_propose(
            moves=((pid, pos),)
        )
        
        assert abs(phi_direct) > 0
        # err = min(abs(prop_energy[0][0] - phi_direct)/abs(phi_direct), abs(prop_energy[0][0] - phi_direct))
        err = abs(prop_energy[0][0] - phi_direct)/abs(phi_direct)
        # print(prop_energy[0][0], phi_direct, abs(prop_energy[0][0] - phi_direct))
        assert err < eps


@pytest.mark.skipif('MPISIZE > 1')
def test_kmc_fmm_nearest_27_2():
    """
    Passes all proposed moves to kmc at once, then checks all outputs
    """
    
    eps = 10.**-5
    L = 12
    R = 3

    N = 50
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
        domain=A.domain, r=R, l=L, boundary_condition='27')
    kmc_fmm.initialise()
    
    # make  some random proposed moves
    order = rng.permutation(range(N))
    prop = []

    for px in range(N):

        propn = rng.randint(1, 8)
        prop.append(
            (
                order[px],
                rng.uniform(low=-0.5*E, high=0.5*E, size=(propn, 3))
            )
        )
    
    # get the energy of the proposed moves
    prop_energy_py = kmc_fmm.test_propose(moves=prop, use_python=True)
    prop_energy_c  = kmc_fmm.test_propose(moves=prop, use_python=False)
    
    # test agains the direct calculation
    for rxi, rx in enumerate(prop):
        pid = rx[0]
        for movi, mov in enumerate(rx[1]):
            
            fmm_phi_py = prop_energy_py[rxi][movi]
            fmm_phi_c = prop_energy_c[rxi][movi]

            assert abs(fmm_phi_py - fmm_phi_c)/abs(fmm_phi_py) < eps


@pytest.mark.skipif('MPISIZE > 1')
def test_kmc_fmm_nearest_27_accept_1():
    """
    Passes all proposed moves to kmc at once, then checks all outputs
    """
    
    eps = 10.**-5
    L = 12
    R = 3

    N = 200
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
        domain=A.domain, r=R, l=L, boundary_condition='27')
    kmc_fmm.initialise()
    
    
    for stepx in range(20):
        # make  some random proposed moves
        order = rng.permutation(range(N))
        prop = []
        
        nmov = 0
        for px in range(N):
            #for px in range(1):

            propn = rng.randint(1, 8)
            #propn = 1
            nmov += propn

            prop.append(
                (
                    order[px],
                    rng.uniform(low=-0.5*E, high=0.5*E, size=(propn, 3))
                )
            )
        
        prop_energy_c1  = kmc_fmm.propose(moves=prop)
        
        # pick a random set of proposals
        pid = rng.randint(0, N-1)
        particle_id = order[pid]

        max_nprop = np.atleast_2d(prop[pid][1]).shape[0]-1
        prop_accept_id = rng.randint(0, max_nprop) if max_nprop > 0 else 0
        prop_accept_energy = prop_energy_c1[pid][prop_accept_id]
        prop_accept_pos = prop[pid][1][prop_accept_id]

        kmc_fmm.test_accept_reinit((particle_id, prop_accept_pos))
        
        rel = abs(prop_accept_energy) if abs(prop_accept_energy) > 1.0 else 1.0

        err = abs(prop_accept_energy - kmc_fmm.energy) / rel
        assert err < 10.**-6
    


@pytest.mark.skipif('MPISIZE > 1')
def test_kmc_fmm_nearest_27_accept_2():

    eps = 10.**-5
    L = 12
    R = 3

    N = 200
    E = 4.
    rc = E/4
    


    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()
    A.npart = N
    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)

    B = state.State()
    B.domain = domain.BaseDomainHalo(extent=(E,E,E))
    B.domain.boundary_condition = domain.BoundaryTypePeriodic()
    B.npart = N
    B.P = data.PositionDat(ncomp=3)
    B.Q = data.ParticleDat(ncomp=1)

    rng  = np.random.RandomState(seed=8657)

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
    
    B.P[:] = A.P.data.copy()
    B.Q[:] = A.Q.data.copy()

    A.scatter_data_from(0)
    B.scatter_data_from(0)

    # create a kmc instance
    kmc_fmmA = KMCFMM(positions=A.P, charges=A.Q, 
        domain=A.domain, r=R, l=L, boundary_condition='27')
    kmc_fmmA.initialise()
    
    kmc_fmmB = KMCFMM(positions=B.P, charges=B.Q, 
        domain=B.domain, r=R, l=L, boundary_condition='27')
    kmc_fmmB.initialise() 
    
    # print("\n arggggg")

    for stepx in range(20):
        # make  some random proposed moves
        order = rng.permutation(range(N))
        prop = []
        
        nmov = 0
        # for px in range(N):
        for px in range(1):

            propn = rng.randint(1, 8)
            # propn = 1
            nmov += propn

            prop.append(
                (
                    order[px],
                    rng.uniform(low=-0.5*E, high=0.5*E, size=(propn, 3))
                )
            )
        
        prop_energyA  = kmc_fmmA.propose(moves=prop)
        prop_energyB  = kmc_fmmB.propose(moves=prop)
        
    
        for pa, pb in zip(prop_energyA, prop_energyB):
            # print("==")
            for pai, pbi in zip(pa, pb):
                rel = abs(pai) if abs(pai) > 1.0 else 1.0
                err = abs(pai - pbi) / rel
                print("--", err)
                assert err < 10.**-5


        # pick a random set of proposals
        pid = rng.randint(0, N-1)
        particle_id = order[pid]

        # max_nprop = np.atleast_2d(prop[pid][1]).shape[0]-1
        # prop_accept_id = rng.randint(0, max_nprop) if max_nprop > 0 else 0

        pid = 0
        prop_accept_id = 0
        prop_accept_energy = prop_energyA[pid][prop_accept_id]
        prop_accept_pos = prop[pid][1][prop_accept_id]
    

        kmc_fmmA.test_accept_reinit((particle_id, prop_accept_pos))
        kmc_fmmB._accept((particle_id, prop_accept_pos))
        
        # print(kmc_fmmA.energy, kmc_fmmB.energy, "----------")

        reeng = abs(kmc_fmmA.energy)
        rel = 1.0 if reeng < 1.0 else reeng
        err = abs(kmc_fmmA.energy - kmc_fmmB.energy) / rel
        print("==========>", err)
        assert err < 10.**-5





