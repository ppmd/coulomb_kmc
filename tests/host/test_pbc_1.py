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

c_double = ctypes.c_double


from coulomb_kmc.common import spherical
from coulomb_kmc.kmc_expansion_tools import LocalExpEval



def test_c_multipole_expansion():
    
    L = 26
    N = 10
    ncomp = 2 * (L**2)

    rng = np.random.RandomState(235243095)

    ExpInst = LocalExpEval(L)
    
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

    ExpInst = LocalExpEval(L)
    
    radii = rng.uniform(0.1, 10, N)
    theta_set = rng.uniform(0.001, pi, N)
    phi_set = rng.uniform(0.001, 2.*pi, N)
    
    for ix in range(N):
        moments = np.array(rng.uniform(size=ncomp), dtype=c_double)

        py_phi = ExpInst.py_compute_phi_local(moments, (radii[ix], theta_set[ix], phi_set[ix]))[0]
        c_phi = ExpInst.compute_phi_local(moments, (radii[ix], theta_set[ix], phi_set[ix]))[0]
        rel = 1.0 if abs(py_phi) < 1.0 else abs(py_phi)
        err = abs(py_phi - c_phi) / rel
        assert err < 10.**-12
        

@pytest.mark.skipif("MPISIZE > 1")
def test_split_concept_1():
    L = 12
    R = 3

    N = 200
    E = 1.
    rc = E/4

    rng = np.random.RandomState(seed=12415)

    ncomp = (L**2)*2
    half_ncomp = (L**2)
    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()
    A.npart = N

    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)

    A.P[:] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,3))
    for px in range(N):
        A.Q[px,0] = (-1.0)**(px+1)
    bias = np.sum(A.Q[:N:, 0])/N
    A.Q[:, 0] -= bias


    A.scatter_data_from(0)


    fmm_pbc = PyFMM(A.domain, N=N, free_space=False, r=R, l=L)
    fmm_27 = PyFMM(A.domain, N=N, free_space='27', r=R, l=L)

    energy_pbc = fmm_pbc(A.P, A.Q)
    energy_27 = fmm_27(A.P, A.Q)
    
    lee = LocalExpEval(L)
    local_dot_coeffs = np.zeros(ncomp, dtype=REAL)
    for px in range(N):
        lee.dot_vec(spherical(tuple(A.P[px, :])), A.Q[px, :], local_dot_coeffs)
    
    L_exp = np.zeros_like(fmm_pbc.tree_parent[1][0, 0, 0, :])
    
    fmm_pbc._lr_mtl_func(fmm_pbc.tree_halo[0][2,2,2,:], L_exp)
    fmm_pbc.dipole_corrector(fmm_pbc.tree_halo[0][2,2,2,:], L_exp)

    lr_energy = 0.5 * np.dot(L_exp, local_dot_coeffs)
    
    err = abs(energy_pbc - (energy_27 + lr_energy)) / abs(energy_pbc)

    assert err < 10.**-14


@pytest.mark.skipif("MPISIZE > 1")
def test_kmc_lr_1():
    """
    Passes all proposed moves to kmc at once, then checks all outputs
    """
    L = 12
    R = 3

    N = 2000
    E = 1.
    rc = E/4

    ncomp = (L**2)*2
    half_ncomp = (L**2)
    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()
    A.npart = N

    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)
    A.PP = data.ParticleDat(ncomp=3)

    A.crr = data.ScalarArray(ncomp=1)

    rng = np.random.RandomState(seed=1234)

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
    
    ExpInst = LocalExpEval(L)
    si = kmc_fmm_self_interaction.FMMSelfInteraction(kmc_fmm.fmm, A.domain, kmc_fmm._bc, ExpInst)

    # make  some random proposed moves
    order = rng.permutation(range(N))
    prop = []
    max_nprop = 1
    for px in range(N):
        #for px in range(1):

        propn = rng.randint(1, 8)
        #propn = 1
        max_nprop = max(max_nprop, propn)

        prop.append(
            (
                order[px],
                rng.uniform(low=-0.5*E, high=0.5*E, size=(propn, 3))
            )
        )
    
    processed_movs = list(kmc_fmm.md.setup_propose(prop))
    correct = np.zeros((N, max_nprop), dtype=c_double)
    matmulv = np.zeros_like(correct)
    si.propose(*tuple(processed_movs + [correct, True]))
    si.propose(*tuple(processed_movs + [matmulv, False]))
    
    for px in range(N):
        #print(correct[px, :])
        #print("--------")
        #print(matmulv[px, :])
        #print("========")
        for mv in range(prop[px][1].shape[0]):
            c = correct[px, mv]
            m = matmulv[px, mv]
            
            rel = 1.0 if abs(c) < 1.0 else abs(c)
            err = abs(c - m)/rel
            assert err < 10.**-15
            



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

    #fmm = PyFMM(B.domain, N=N, free_space='27', r=kmc_fmm.fmm.R, l=kmc_fmm.fmm.L)
    #print("WRONG FMM")
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
@pytest.mark.parametrize("R", (3, 4, 5))
def test_kmc_fmm_pbc_3(R):
    """
    Passes all proposed moves to kmc at once, then checks all outputs
    """
    
    eps = 10.**-5
    L = 12

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
        domain=A.domain, r=R, l=L, boundary_condition='pbc')

 

    # make  some random proposed moves
    order = rng.permutation(range(N))
    prop = []
    nmov = 0
    for px in range(N):

        propn = rng.randint(1, 8)
        nmov += propn

        prop.append(
            (
                order[px],
                rng.uniform(low=-0.5*E, high=0.5*E, size=(propn, 3))
            )
        )
    
    #import cProfile
    #pr = cProfile.Profile()
    #pr.enable()
    
    kmc_fmm.initialise()

    t0 = time.time()
    prop_energy_c  = kmc_fmm.test_propose(moves=prop, use_python=False)
    t1 = time.time()
    #pr.disable()
    #pr.dump_stats('/tmp/propose.prof')
    #print("C :", t1 - t0, N, nmov)
    #common.print_profile()
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


@pytest.mark.skipif('MPISIZE > 1')
def test_kmc_fmm_pbc_accept_1():
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
        domain=A.domain, r=R, l=L, boundary_condition='pbc')
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
        assert err < 10.**-5


@pytest.mark.skipif('MPISIZE > 1')
def test_kmc_fmm_pbc_accept_1_5():

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
        domain=A.domain, r=R, l=L, boundary_condition='pbc')
    kmc_fmmA.initialise()
    
    kmc_fmmB = KMCFMM(positions=B.P, charges=B.Q, 
        domain=B.domain, r=R, l=L, boundary_condition='pbc')
    kmc_fmmB.initialise() 
    
    # print("\n arggggg")

    for stepx in range(4):
        # make  some random proposed moves
        order = rng.permutation(range(N))
        prop = []
        
        nmov = 0
        for px in range(N):
        #for px in range(1):

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
                #print("--", err)
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
    

        movedata = np.zeros(10, dtype=ctypes.c_int64)
        realdata = movedata[:7].view(dtype=ctypes.c_double)

        realdata[0:3:] = A.P[particle_id, :]
        realdata[3:6:] = prop_accept_pos
        realdata[6] = A.Q[particle_id, 0]
        movedata[7] = A._fmm_cell[particle_id, 0]
        movedata[8] = kmc_fmmA._get_lin_cell(prop_accept_pos)


        kmc_fmmA.kmco._accept_py(movedata)
        kmc_fmmB.kmco._accept(movedata)

        lsd = kmc_fmmA.kmco.local_store_dims
        lsdi = (range(lsd[0]), range(lsd[1]), range(lsd[2]))

        for lsx in product(*lsdi):
            py = kmc_fmmA.kmco.local_expansions[lsx[0], lsx[1], lsx[2], :]
            cc = kmc_fmmB.kmco.local_expansions[lsx[0], lsx[1], lsx[2], :]
            
            rel = np.linalg.norm(py, np.inf)
            rel = 1.0 if rel < 1.0 else rel
            err = np.linalg.norm(py - cc, np.inf) / rel

            assert err<10.**-13



@pytest.mark.skipif('MPISIZE > 1')
def test_kmc_fmm_pbc_accept_2():

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
        domain=A.domain, r=R, l=L, boundary_condition='pbc')
    kmc_fmmA.initialise()
    
    kmc_fmmB = KMCFMM(positions=B.P, charges=B.Q, 
        domain=B.domain, r=R, l=L, boundary_condition='pbc')
    kmc_fmmB.initialise() 
    
    # print("\n arggggg")

    for stepx in range(20):
        # make  some random proposed moves
        order = rng.permutation(range(N))
        prop = []
        
        nmov = 0
        # for px in range(N):
        for px in range(20):

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
                #print("--", err)
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
        #print("==========>", err)
        assert err < 10.**-5


    
