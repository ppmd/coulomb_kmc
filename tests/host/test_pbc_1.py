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
def test_kmc_fmm_pbc_1():
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
    B.F = data.ParticleDat(ncomp=3)


    rng = np.random.RandomState(seed=1234)

    if N == 4:

        A.P[0,:] = ( 0.80,  0.20, 0.0)
        A.P[1,:] = (-0.80,  0.20, 0.0)
        A.P[2,:] = (-0.80, -0.20, 0.0)
        A.P[3,:] = ( 0.80, -0.20, 0.0)
        
        q = 2.

        A.Q[0,0] = -q
        A.Q[1,0] = q
        A.Q[2,0] = -q
        A.Q[3,0] = q
    
    elif N == 1:

        A.P[0,:] = (  0.00,  0.00,  0.00)
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
    
    bcs = (False, 'pbc')
 
    kmc_fmm = KMCFMM(positions=A.P, charges=A.Q, 
        domain=A.domain, r=R, l=L, boundary_condition=bcs[1])
    kmc_fmm.initialise()

    fmm = PyFMM(B.domain, N=N, free_space=bcs[0], r=kmc_fmm.fmm.R, l=kmc_fmm.fmm.L)
    #ewald = EwaldOrthoganalHalf(domain=B.domain, real_cutoff=rc, shared_memory='omp', eps=10.**-8)


    def _direct():
        _phi_direct = fmm(positions=B.P, charges=B.Q)
        #_phi_direct2 = ewald(positions=B.P, charges=B.Q, forces=B.F)
        #print("---")
        #print(_phi_direct, _phi_direct2, abs(_phi_direct - _phi_direct2))
        #print("---")
        return _phi_direct
    
    phi_direct = _direct()


    for rx in range(N):
        #pid = 0
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
def test_kmc_fmm_pbc_2():
    """
    Tests proposed moves one by one against direct calculation.
    Considers the primary image and the nearest 27 neighbours.
    """

    eps = 10.**-5
    L = 14
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
    B.F = data.ParticleDat(ncomp=3)


    rng = np.random.RandomState(seed=1234)

    if N == 4:

        A.P[0,:] = ( 0.80,  0.20, 0.0)
        A.P[1,:] = (-0.80,  0.20, 0.0)
        A.P[2,:] = (-0.80, -0.20, 0.0)
        A.P[3,:] = ( 0.80, -0.20, 0.0)
        
        q = 2.

        A.Q[0,0] = -q
        A.Q[1,0] = q
        A.Q[2,0] = -q
        A.Q[3,0] = q
    
    elif N == 1:

        A.P[0,:] = (  0.00,  0.00,  0.00)
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
    
    bcs = (False, 'pbc')
 
    kmc_fmm = KMCFMM(positions=A.P, charges=A.Q, 
        domain=A.domain, r=R, l=L, boundary_condition=bcs[1])
    kmc_fmm.initialise()

    fmm = PyFMM(B.domain, N=N, free_space=bcs[0], r=kmc_fmm.fmm.R, l=kmc_fmm.fmm.L)

    def _direct():
        _phi_direct = fmm(positions=B.P, charges=B.Q)
        return _phi_direct
    
    phi_direct = _direct()

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
    prop_energy = kmc_fmm.test_propose(moves=prop)
    
    # test agains the direct calculation
    for rxi, rx in enumerate(prop):
        pid = rx[0]
        for movi, mov in enumerate(rx[1]):
            
            B.P[pid, :] = mov

            phi_direct = _direct()
            
            B.P[pid, :] = A.P[pid, :]
        
            assert abs(phi_direct) > 0
            
            fmm_phi = prop_energy[rxi][movi]

            assert abs(fmm_phi - phi_direct)/abs(phi_direct) < eps




