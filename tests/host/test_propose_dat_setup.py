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

@pytest.mark.skipif("MPISIZE > 1")
@pytest.mark.parametrize("param_boundary", ('free_space', 'pbc', '27'))
@pytest.mark.parametrize("R", (3, 4, 5))
def test_kmc_fmm_dat_setup_prop_1(param_boundary, R):

    L = 4

    N = 100
    E = 4.123
    rc = E/4
    M = 6

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()
    A.npart = N
    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)
    A.prop_masks = data.ParticleDat(ncomp=M, dtype=INT64)
    A.prop_positions = data.ParticleDat(ncomp=M*3)
    A.prop_diffs = data.ParticleDat(ncomp=M)
    A.sites = data.ParticleDat(ncomp=1, dtype=INT64)

    B = state.State()
    B.domain = domain.BaseDomainHalo(extent=(E,E,E))
    B.domain.boundary_condition = domain.BoundaryTypePeriodic()
    B.npart = N
    B.P = data.PositionDat(ncomp=3)
    B.Q = data.ParticleDat(ncomp=1)

    rng  = np.random.RandomState(seed=8657)

    site_max_counts = data.ScalarArray(ncomp=8, dtype=INT64)
    site_max_counts[:] = rng.randint(0, 10, size=8)

    A.P[:] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,3))
    for px in range(N):
        A.Q[px,0] = (-1.0)**(px+1)
    bias = np.sum(A.Q[:N:, 0])/N
    A.Q[:, 0] -= bias
    A.sites[:, 0] = rng.randint(0, 8, size=N)

    
    B.P[:] = A.P.data.copy()
    B.Q[:] = A.Q.data.copy()

    A.scatter_data_from(0)
    B.scatter_data_from(0)

    # create a kmc instance
    kmc_fmmA = KMCFMM(positions=A.P, charges=A.Q, 
        domain=A.domain, r=R, l=L, boundary_condition=param_boundary)
    kmc_fmmA.initialise()
    
    kmc_fmmB = KMCFMM(positions=B.P, charges=B.Q, 
        domain=B.domain, r=R, l=L, boundary_condition=param_boundary)
    kmc_fmmB.initialise() 
    
    # print("\n arggggg")

    # for stepx in range(20):
    prop = []
    
    nmov = 0
    for px in range(A.npart_local):
        tmp = []
        masks = np.zeros(M)
        masks[:site_max_counts[A.sites[px,0]]:] = 1
        masks = rng.permutation(masks)

        for propx in range(M):
            mask = masks[propx]
            prop_pos = rng.uniform(low=-0.5*E, high=0.5*E, size=3)
            A.prop_masks[px, propx] = mask
            A.prop_positions[px, propx*3:propx*3+3:] = prop_pos
            
            if mask > 0:
                tmp.append(list(prop_pos))
                nmov += 1
        if len(tmp) > 0:
            prop.append((px, np.array(tmp)))

    correct = kmc_fmmB.md.setup_propose(prop)
    to_test =  kmc_fmmA.md.setup_propose_with_dats(site_max_counts, A.sites,
        A.prop_positions, A.prop_masks, A.prop_diffs)
    

    assert to_test[0] == correct[0]
    assert to_test[1] == correct[1]
    
    total_movs = correct[0]
    num_particles = correct[1]

    tind = 0
    for px in range(A.npart_local):
        for prop_pos in range(M):
            if A.prop_masks[px, prop_pos] > 0:
                assert to_test[2]['rate_location'][tind, 0] == px*M+prop_pos
                tind += 1


    err = np.linalg.norm(
        correct[2]['exclusive_sum'][:num_particles:, :].ravel() - \
        to_test[2]['exclusive_sum'][:num_particles:, :].ravel(),
        np.inf
    )
    assert err < 10.**-15
   
    err = np.linalg.norm(
        correct[2]['old_positions'][:num_particles:, :].ravel() - \
        to_test[2]['old_positions'][:num_particles:, :].ravel(),
        np.inf
    )
    assert err < 10.**-15

    err = np.linalg.norm(
        correct[2]['old_charges'][:num_particles:, :].ravel() - \
        to_test[2]['old_charges'][:num_particles:, :].ravel(),
        np.inf
    )
    assert err < 10.**-15

    err = np.linalg.norm(
        correct[2]['old_ids'][:num_particles:, :].ravel() - \
        to_test[2]['old_ids'][:num_particles:, :].ravel(),
        np.inf
    )
    assert err < 10.**-15

    err = np.linalg.norm(
        correct[2]['new_positions'][:total_movs:, :].ravel() - \
        to_test[2]['new_positions'][:total_movs:, :].ravel(),
        np.inf
    )
    assert err < 10.**-15

    err = np.linalg.norm(
        correct[2]['new_charges'][:total_movs:, :].ravel() - \
        to_test[2]['new_charges'][:total_movs:, :].ravel(),
        np.inf
    )
    assert err < 10.**-15

    err = np.linalg.norm(
        correct[2]['new_ids'][:total_movs:, :].ravel() - \
        to_test[2]['new_ids'][:total_movs:, :].ravel(),
        np.inf
    )
    assert err < 10.**-15

    
    err = np.linalg.norm(
        correct[2]['old_fmm_cells'][:total_movs:, :].ravel() - \
        to_test[2]['old_fmm_cells'][:total_movs:, :].ravel(),
        np.inf
    )
    assert err < 10.**-15

    
    err = np.linalg.norm(
        correct[2]['new_fmm_cells'][:total_movs:, :].ravel() - \
        to_test[2]['new_fmm_cells'][:total_movs:, :].ravel(),
        np.inf
    )
    assert err < 10.**-15

    
    err = np.linalg.norm(
        correct[2]['new_shifted_positions'][:total_movs:, :].ravel() - \
        to_test[2]['new_shifted_positions'][:total_movs:, :].ravel(),
        np.inf
    )
    assert err < 10.**-15

    kmc_fmmA.free()
    kmc_fmmB.free()





@pytest.mark.parametrize("param_boundary", ('free_space', 'pbc', '27'))
@pytest.mark.parametrize("R", (3, 4, 5))
def test_kmc_fmm_dat_setup_prop_2(param_boundary, R):

    L = 4

    N = 100
    E = 4.
    rc = E/4
    M = 6

    max_move = E*0.255
    max_move_setup = 1.01 * max_move

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()
    A.npart = N
    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)
    A.prop_masks = data.ParticleDat(ncomp=M, dtype=INT64)
    A.prop_positions = data.ParticleDat(ncomp=M*3)
    A.prop_diffs = data.ParticleDat(ncomp=M)
    A.sites = data.ParticleDat(ncomp=1, dtype=INT64)

    B = state.State()
    B.domain = domain.BaseDomainHalo(extent=(E,E,E))
    B.domain.boundary_condition = domain.BoundaryTypePeriodic()
    B.npart = N
    B.P = data.PositionDat(ncomp=3)
    B.Q = data.ParticleDat(ncomp=1)

    rng  = np.random.RandomState(seed=8657*R)

    site_max_counts = data.ScalarArray(ncomp=8, dtype=INT64)
    site_max_counts[:] = rng.randint(0, 10, size=8)

    A.P[:] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,3))
    for px in range(N):
        A.Q[px,0] = (-1.0)**(px+1)
    bias = np.sum(A.Q[:N:, 0])/N
    A.Q[:, 0] -= bias
    A.sites[:, 0] = rng.randint(0, 8, size=N)
    
    A.P[0,:] = ( 0.499999*E, 0, 0)
    A.P[1,:] = (-0.499999*E, 0, 0)
    A.P[2,:] = (0,  0.499999*E, 0)
    A.P[3,:] = (0, -0.499999*E, 0)
    A.P[4,:] = (0, 0,  0.499999*E)
    A.P[5,:] = (0, 0, -0.499999*E)    

    
    B.P[:] = A.P.data.copy()
    B.Q[:] = A.Q.data.copy()

    A.scatter_data_from(0)
    B.scatter_data_from(0)

    # create a kmc instance
    kmc_fmmA = KMCFMM(positions=A.P, charges=A.Q, 
        domain=A.domain, r=R, l=L, boundary_condition=param_boundary, max_move=max_move_setup)
    kmc_fmmA.initialise()
    
    
    # print("\n arggggg")

    # for stepx in range(20):
    prop = []
    
    nmov = 0
    # for px in range(N):
    for px in range(A.npart_local):
        tmp = []
        masks = np.zeros(M)
        masks[:site_max_counts[A.sites[px,0]]:] = 1
        masks = rng.permutation(masks)

        for propx in range(M):
            mask = masks[propx]


            offset_vector = rng.uniform(low=-1.0, high=1.0, size=3)
            offset_vector /= np.linalg.norm(offset_vector)
            assert abs(np.linalg.norm(offset_vector) - 1.0) < 10.**-15
            offset_size = rng.uniform(low=0.01 * max_move, high=0.99*max_move)
            offset_vector *= offset_size
            pos = A.P[px, :].copy() + offset_vector

            
            if param_boundary in ('pbc', '27'):
                for dimx in range(3):
                    pos[dimx] = np.fmod(pos[dimx] + 1.5*E, E) - 0.5*E
            elif param_boundary == 'free_space':
                flush = 0.5
                for dimx in range(3):
                    pos[dimx] = min(pos[dimx], E*flush)
                    pos[dimx] = max(pos[dimx], E*(-flush))
            else:
                raise RuntimeError('Bad boundary condition')

            prop_pos = pos

            A.prop_masks[px, propx] = mask
            A.prop_positions[px, propx*3:propx*3+3:] = prop_pos
            
            if mask > 0:
                tmp.append(list(prop_pos))
                nmov += 1
        if len(tmp) > 0:
            prop.append((px, np.array(tmp)))

    cf = kmc_fmmA.md.setup_propose(prop)
    
    # we have to resue the same kmc to avoid race conditin in the gids assigned by
    # the kmc instance.
    # cf is a reference, we need to copy the values
    correct = [cf[0], cf[1], dict()]
    for kx in cf[2].keys():
        correct[2][kx] = cf[2][kx].copy()


    to_test =  kmc_fmmA.md.setup_propose_with_dats(site_max_counts, A.sites,
        A.prop_positions, A.prop_masks, A.prop_diffs)
    

    assert to_test[0] == correct[0]
    assert to_test[1] == correct[1]
    
    total_movs = correct[0]
    num_particles = correct[1]

    tind = 0
    for px in range(A.npart_local):
        for prop_pos in range(M):
            if A.prop_masks[px, prop_pos] > 0:
                assert to_test[2]['rate_location'][tind, 0] == px*M+prop_pos
                tind += 1

    if A.npart_local > 0:
        err = np.linalg.norm(
            correct[2]['exclusive_sum'][:num_particles:, :].ravel() - \
            to_test[2]['exclusive_sum'][:num_particles:, :].ravel(),
            np.inf
        )
        assert err < 10.**-15
       
        err = np.linalg.norm(
            correct[2]['old_positions'][:num_particles:, :].ravel() - \
            to_test[2]['old_positions'][:num_particles:, :].ravel(),
            np.inf
        )
        assert err < 10.**-15

        err = np.linalg.norm(
            correct[2]['old_charges'][:num_particles:, :].ravel() - \
            to_test[2]['old_charges'][:num_particles:, :].ravel(),
            np.inf
        )
        assert err < 10.**-15

        err = np.linalg.norm(
            correct[2]['old_ids'][:num_particles:, :].ravel() - \
            to_test[2]['old_ids'][:num_particles:, :].ravel(),
            np.inf
        )
        assert err < 10.**-15

        err = np.linalg.norm(
            correct[2]['new_positions'][:total_movs:, :].ravel() - \
            to_test[2]['new_positions'][:total_movs:, :].ravel(),
            np.inf
        )
        assert err < 10.**-15

        err = np.linalg.norm(
            correct[2]['new_charges'][:total_movs:, :].ravel() - \
            to_test[2]['new_charges'][:total_movs:, :].ravel(),
            np.inf
        )
        assert err < 10.**-15

        err = np.linalg.norm(
            correct[2]['new_ids'][:total_movs:, :].ravel() - \
            to_test[2]['new_ids'][:total_movs:, :].ravel(),
            np.inf
        )
        assert err < 10.**-15

        
        err = np.linalg.norm(
            correct[2]['old_fmm_cells'][:total_movs:, :].ravel() - \
            to_test[2]['old_fmm_cells'][:total_movs:, :].ravel(),
            np.inf
        )
        assert err < 10.**-15

        
        err = np.linalg.norm(
            correct[2]['new_fmm_cells'][:total_movs:, :].ravel() - \
            to_test[2]['new_fmm_cells'][:total_movs:, :].ravel(),
            np.inf
        )
        assert err < 10.**-15

        
        err = np.linalg.norm(
            correct[2]['new_shifted_positions'][:total_movs:, :].ravel() - \
            to_test[2]['new_shifted_positions'][:total_movs:, :].ravel(),
            np.inf
        )
        assert err < 10.**-15

    kmc_fmmA.free()





















@pytest.mark.skipif("MPISIZE > 1")
@pytest.mark.parametrize("param_boundary", ('free_space', 'pbc', '27'))
@pytest.mark.parametrize("R", (3, 4, 5))
def test_kmc_fmm_dat_setup_prop_3(param_boundary, R):

    L = 8

    N = 200
    E = 4.
    rc = E/4
    M = 8

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()
    A.npart = N
    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)
    A.prop_masks = data.ParticleDat(ncomp=M, dtype=INT64)
    A.prop_positions = data.ParticleDat(ncomp=M*3)
    A.prop_diffs = data.ParticleDat(ncomp=M)
    A.sites = data.ParticleDat(ncomp=1, dtype=INT64)

    B = state.State()
    B.domain = domain.BaseDomainHalo(extent=(E,E,E))
    B.domain.boundary_condition = domain.BoundaryTypePeriodic()
    B.npart = N
    B.P = data.PositionDat(ncomp=3)
    B.Q = data.ParticleDat(ncomp=1)

    rng  = np.random.RandomState(seed=8657*R)

    site_max_counts = data.ScalarArray(ncomp=8, dtype=INT64)
    site_max_counts[:] = rng.randint(0, 10, size=8)

    A.P[:] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,3))
    for px in range(N):
        A.Q[px,0] = (-1.0)**(px+1)
    bias = np.sum(A.Q[:N:, 0])/N
    A.Q[:, 0] -= bias
    A.sites[:, 0] = rng.randint(0, 8, size=N)

    
    B.P[:] = A.P.data.copy()
    B.Q[:] = A.Q.data.copy()

    A.scatter_data_from(0)
    B.scatter_data_from(0)

    # create a kmc instance
    kmc_fmmA = KMCFMM(positions=A.P, charges=A.Q, 
        domain=A.domain, r=R, l=L, boundary_condition=param_boundary)
    kmc_fmmA.initialise()
    
    kmc_fmmB = KMCFMM(positions=B.P, charges=B.Q, 
        domain=B.domain, r=R, l=L, boundary_condition=param_boundary)
    kmc_fmmB.initialise() 
    
    # print("\n arggggg")

    # for stepx in range(20):

    for testx in range(4):
        prop = []
        nmov = 0
        for px in range(N):
            tmp = []
            masks = np.zeros(M)
            masks[:site_max_counts[A.sites[px,0]]:] = 1
            masks = rng.permutation(masks)

            for propx in range(M):
                mask = masks[propx]
                prop_pos = rng.uniform(low=-0.5*E, high=0.5*E, size=3)
                A.prop_masks[px, propx] = mask
                A.prop_positions[px, propx*3:propx*3+3:] = prop_pos
                
                if mask > 0:
                    tmp.append(list(prop_pos))
                    nmov += 1
            if len(tmp) > 0:
                prop.append((px, np.array(tmp)))

        t0 = time.time()
        correct = kmc_fmmB.propose(prop)
        t1= time.time()
        to_test =  kmc_fmmA.propose_with_dats(site_max_counts, A.sites,
            A.prop_positions, A.prop_masks, A.prop_diffs, diff=True)
        t2 = time.time()


        for propi, propx in enumerate(prop):
            pid = propx[0]
            movs = propx[1]
            found_movs = 0
            for pmi in range(M):
                if A.prop_masks[pid, pmi] > 0:
                    correct_energy = correct[propi][found_movs]
                    to_test_energy = A.prop_diffs[pid, pmi] + kmc_fmmA.energy
                    
                    rel = 1.0 if abs(correct_energy) < 1 else abs(correct_energy)
                    err = abs(correct_energy - to_test_energy) / rel
                    
                    assert err < 10**-12

                    found_movs += 1
            

    # opt.print_profile()
    kmc_fmmA.free()
    kmc_fmmB.free()






@pytest.mark.skipif("MPISIZE > 1")
@pytest.mark.parametrize("param_boundary", (
    ('free_space', True), ('pbc', False), ('27', '27')
))
@pytest.mark.parametrize("R", (3, 4, 5))
def test_kmc_fmm_realistic_1(param_boundary, R):
    
    L = 12
    
    N = 200
    E = 4.
    rc = E/4
    M = 4
    N_steps = 10
    
    tol_ewald = 10.**-4
    tol_kmc = 10.**-5

    EWALD = True if param_boundary[0] == 'pbc' else False


    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()
    A.npart = N
    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)
    A.prop_masks = data.ParticleDat(ncomp=M, dtype=INT64)
    A.prop_positions = data.ParticleDat(ncomp=M*3)
    A.prop_diffs = data.ParticleDat(ncomp=M)
    A.sites = data.ParticleDat(ncomp=1, dtype=INT64)

    B = state.State()
    B.domain = domain.BaseDomainHalo(extent=(E,E,E))
    B.domain.boundary_condition = domain.BoundaryTypePeriodic()
    B.npart = N
    B.P = data.PositionDat(ncomp=3)
    B.Q = data.ParticleDat(ncomp=1)
    B.F = data.ParticleDat(ncomp=3)

    rng  = np.random.RandomState(seed=8657)

    site_max_counts = data.ScalarArray(ncomp=8, dtype=INT64)
    site_max_counts[:] = rng.randint(0, 10, size=8)

    A.P[:] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,3))
    for px in range(N):
        A.Q[px,0] = (-1.0)**(px+1)
    bias = np.sum(A.Q[:N:, 0])/N
    A.Q[:, 0] -= bias
    A.sites[:, 0] = rng.randint(0, 8, size=N)

    
    B.P[:] = A.P.data.copy()
    B.Q[:] = A.Q.data.copy()

    A.scatter_data_from(0)
    B.scatter_data_from(0)

    # create a kmc instance
    kmc_fmmA = KMCFMM(positions=A.P, charges=A.Q, 
        domain=A.domain, r=R, l=L, boundary_condition=param_boundary[0])
    kmc_fmmA.initialise()
    
    # no point using more than 3 levels here
    fmm = PyFMM(B.domain, N=N, free_space=param_boundary[1], r=3, l=L)
    
    if EWALD:
        ewald = EwaldOrthoganalHalf(domain=B.domain, real_cutoff=rc, shared_memory='omp', eps=10.**-8)


    def _direct():
        _phi_direct = fmm(positions=B.P, charges=B.Q)

        if EWALD:
            _phi_direct2 = ewald(positions=B.P, charges=B.Q, forces=B.F)
        
            rel = abs(_phi_direct2)
            rel = rel if rel > 0 else 1.0
            err = abs(_phi_direct - _phi_direct2) / rel

            assert err < tol_ewald

        return _phi_direct
    
    
    def _check_system_energy():

        init_energy = _direct()
        rel = abs(init_energy)
        rel = rel if rel > 0 else 1.0
        err = abs(init_energy - kmc_fmmA.energy) / rel

        assert err < tol_kmc


    _check_system_energy()
    
    for testx in range(N_steps):


        prop = []
        nmov = 0
        for px in range(N):
            tmp = []
            masks = np.zeros(M)
            masks[:site_max_counts[A.sites[px,0]]:] = 1
            masks = rng.permutation(masks)

            for propx in range(M):
                mask = masks[propx]
                prop_pos = rng.uniform(low=-0.5*E, high=0.5*E, size=3)
                A.prop_masks[px, propx] = mask
                A.prop_positions[px, propx*3:propx*3+3:] = prop_pos
                
                if mask > 0:
                    tmp.append(list(prop_pos))
                    nmov += 1
            if len(tmp) > 0:
                prop.append((px, np.array(tmp)))



        to_test =  kmc_fmmA.propose_with_dats(site_max_counts, A.sites,
            A.prop_positions, A.prop_masks, A.prop_diffs, diff=True)


        for propi, propx in enumerate(prop):
            pid = propx[0]
            movs = propx[1]
            found_movs = 0
            for pmi in range(M):
                if A.prop_masks[pid, pmi] > 0:
                    mov = A.prop_positions[pid, pmi*3:(pmi+1)*3:]

                    B.P[pid, :] = mov
                    correct_energy = _direct()
                    B.P[pid, :] = A.P[pid, :]


                    to_test_energy = A.prop_diffs[pid, pmi] + kmc_fmmA.energy
                    
                    rel = 1.0 if abs(correct_energy) < 1 else abs(correct_energy)
                    err = abs(correct_energy - to_test_energy) / rel
                    
                    assert err < tol_kmc

                    found_movs += 1

        # pick random accept

        pid = rng.randint(0, A.npart_local-1)
        pmi = rng.randint(0, M-1)
        mov = A.prop_positions[pid, pmi*3:(pmi+1)*3:]
        
        kmc_fmmA.accept((pid, mov))
        B.P[pid, :] = mov


        _check_system_energy()


    # opt.print_profile()


    fmm.free()
    kmc_fmmA.free()




@pytest.mark.skipif("MPISIZE > 1")
@pytest.mark.parametrize("param_boundary", (
    ('free_space', True), ('pbc', False)
))
@pytest.mark.parametrize("R", (3, 4, 5))
def test_kmc_fmm_realistic_2(param_boundary, R):
    
    L = 12
    
    N = 100
    E = 31.
    rc = E/4
    M = 4
    N_steps = 10
    
    tol_ewald = 10.**-3
    tol_kmc = 10.**-5


    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()
    A.npart = N
    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)
    A.prop_masks = data.ParticleDat(ncomp=M, dtype=INT64)
    A.prop_positions = data.ParticleDat(ncomp=M*3)
    A.prop_diffs = data.ParticleDat(ncomp=M)
    A.sites = data.ParticleDat(ncomp=1, dtype=INT64)

    B = state.State()
    B.domain = domain.BaseDomainHalo(extent=(E,E,E))
    B.domain.boundary_condition = domain.BoundaryTypePeriodic()
    B.npart = N
    B.P = data.PositionDat(ncomp=3)
    B.Q = data.ParticleDat(ncomp=1)
    B.F = data.ParticleDat(ncomp=3)

    rng  = np.random.RandomState(seed=817)

    site_max_counts = data.ScalarArray(ncomp=8, dtype=INT64)
    site_max_counts[:] = rng.randint(0, 10, size=8)

    A.P[:] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,3))
    for px in range(N):
        A.Q[px,0] = (-10.0)**(px+1)
    bias = np.sum(A.Q[:N:, 0])/N
    A.Q[:, 0] -= bias
    A.sites[:, 0] = rng.randint(0, 8, size=N)

    
    B.P[:] = A.P.data.copy()
    B.Q[:] = A.Q.data.copy()

    A.scatter_data_from(0)
    B.scatter_data_from(0)

    # create a kmc instance
    kmc_fmmA = KMCFMM(positions=A.P, charges=A.Q, 
        domain=A.domain, r=R, l=L, boundary_condition=param_boundary[0])
    kmc_fmmA.initialise()
    
    # no point using more than 3 levels here
    fmm = PyFMM(B.domain, N=N, free_space=param_boundary[1], r=3, l=L)
    
    def _direct():
        _phi_direct_fmm = fmm(positions=B.P, charges=B.Q)
        if param_boundary[0] == 'free_space':
            _phi_direct = 0.0
            # compute phi from image and surrounding 26 cells
            for ix in range(N):
                phi_part = 0.0
                for jx in range(ix+1, N):
                    rij = np.linalg.norm(B.P[jx,:] - B.P[ix,:])
                    _phi_direct += B.Q[ix, 0] * B.Q[jx, 0] /rij

            rel = abs(_phi_direct)
            rel = 1.0 if rel == 0 else rel
            err =  abs(_phi_direct - _phi_direct_fmm) / rel
            if err > tol_ewald:
                print(err)
            #assert err < tol_ewald

        return _phi_direct_fmm
    
    
    def _check_system_energy():

        init_energy = _direct()
        rel = abs(init_energy)
        rel = rel if rel > 0 else 1.0
        err = abs(init_energy - kmc_fmmA.energy) / rel

        assert err < tol_kmc


    _check_system_energy()
    
    for testx in range(N_steps):


        prop = []
        nmov = 0
        for px in range(N):
            tmp = []
            masks = np.zeros(M)
            masks[:site_max_counts[A.sites[px,0]]:] = 1
            masks = rng.permutation(masks)

            for propx in range(M):
                mask = masks[propx]
                prop_pos = rng.uniform(low=-0.5*E, high=0.5*E, size=3)
                A.prop_masks[px, propx] = mask
                A.prop_positions[px, propx*3:propx*3+3:] = prop_pos
                
                if mask > 0:
                    tmp.append(list(prop_pos))
                    nmov += 1
            if len(tmp) > 0:
                prop.append((px, np.array(tmp)))



        to_test =  kmc_fmmA.propose_with_dats(site_max_counts, A.sites,
            A.prop_positions, A.prop_masks, A.prop_diffs, diff=True)


        for propi, propx in enumerate(prop):
            pid = propx[0]
            movs = propx[1]
            found_movs = 0
            for pmi in range(M):
                if A.prop_masks[pid, pmi] > 0:
                    mov = A.prop_positions[pid, pmi*3:(pmi+1)*3:]

                    B.P[pid, :] = mov
                    correct_energy = _direct()
                    B.P[pid, :] = A.P[pid, :]


                    to_test_energy = A.prop_diffs[pid, pmi] + kmc_fmmA.energy
                    
                    rel = 1.0 if abs(correct_energy) < 1 else abs(correct_energy)
                    err = abs(correct_energy - to_test_energy) / rel
                    
                    assert err < tol_kmc

                    found_movs += 1

        # pick random accept

        pid = rng.randint(0, A.npart_local-1)
        pmi = rng.randint(0, M-1)
        mov = A.prop_positions[pid, pmi*3:(pmi+1)*3:]
        
        kmc_fmmA.accept((pid, mov))
        B.P[pid, :] = mov


        _check_system_energy()


    # opt.print_profile()


    fmm.free()
    kmc_fmmA.free()









