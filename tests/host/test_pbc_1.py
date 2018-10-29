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

c_double = ctypes.c_double


def test_c_multipole_expansion():
    
    L = 26
    N = 10
    ncomp = 2 * (L**2)

    rng = np.random.RandomState(235243095)

    ExpInst = kmc_fmm_common.LocalExpEval(L)
    
    radii = rng.uniform(0.1, 10, N)
    theta_set = rng.uniform(0.001, pi, N)
    phi_set = rng.uniform(0.001, 2.*pi, N)
    
    
    for ix in range(N):
        arr_py = np.zeros(ncomp, dtype=c_double)
        arr_c  = np.zeros(ncomp, dtype=c_double)

        ExpInst.py_multipole_exp((radii[ix], theta_set[ix], phi_set[ix]), 1.0, arr_py)
        ExpInst.multipole_exp((radii[ix], theta_set[ix], phi_set[ix]), 1.0, arr_c)
        
        for cx in range(ncomp):
            rel = 1.0 if abs(arr_py[cx]) < 1.0 else abs(arr_py[cx])
            err = abs(arr_py[cx] - arr_c[cx]) / rel
            assert err < 10.**-12


def test_c_local_expansion_eval():
    
    L = 26
    N = 10
    ncomp = 2 * (L**2)

    rng = np.random.RandomState(235243095)

    ExpInst = kmc_fmm_common.LocalExpEval(L)
    
    radii = rng.uniform(0.1, 10, N)
    theta_set = rng.uniform(0.001, pi, N)
    phi_set = rng.uniform(0.001, 2.*pi, N)
    
    print('\n')
    for ix in range(N):
        moments = np.array(rng.uniform(size=ncomp), dtype=c_double)

        py_phi = ExpInst.py_compute_phi_local(moments, (radii[ix], theta_set[ix], phi_set[ix]))[0]
        c_phi = ExpInst.compute_phi_local(moments, (radii[ix], theta_set[ix], phi_set[ix]))[0]
        rel = 1.0 if abs(py_phi) < 1.0 else abs(py_phi)
        err = abs(py_phi - c_phi) / rel
        assert err < 10.**-12
        











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


@pytest.mark.skipif('MPISIZE > 1')
def test_kmc_fmm_pbc_3():
    """
    Passes all proposed moves to kmc at once, then checks all outputs
    """
    
    eps = 10.**-5
    L = 12
    R = 3

    N = 2000
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
        domain=A.domain, r=R, l=L, boundary_condition='pbc')
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
    
    import cProfile
    pr = cProfile.Profile()
    pr.enable()
    t0 = time.time()
    prop_energy_c  = kmc_fmm.test_propose(moves=prop, use_python=False)
    t1 = time.time()
    pr.disable()
    pr.dump_stats('/tmp/propose.prof')
    print("C :", t1 - t0)
    common.print_profile()
    # get the energy of the proposed moves


    prop_energy_py = kmc_fmm.test_propose(moves=prop, use_python=True)
    
    # test agains the direct calculation
    for rxi, rx in enumerate(prop):
        pid = rx[0]
        for movi, mov in enumerate(rx[1]):
            
            fmm_phi_py = prop_energy_py[rxi][movi]
            fmm_phi_c = prop_energy_c[rxi][movi]

            assert abs(fmm_phi_py - fmm_phi_c)/abs(fmm_phi_py) < eps





def test_kmc_fmm_pbc_parallel_1():

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
    

    return
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



