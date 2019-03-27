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


from kmc_test_common import *


@pytest.mark.parametrize("R", (3, 4, 5))
def test_kmc_fmm_free_space_propose_1(R):
    """
    Tests proposed moves one by one against direct calculation.
    """

    if R < 4 and MPISIZE > 8:
        return

    eps = 10.**-5
    L = 12

    N = 100
    E = 2*3.1415
    rc = E/4

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()
    A.npart = N

    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)
    A.GID = data.ParticleDat(ncomp=1, dtype=INT64)

    A.crr = data.ScalarArray(ncomp=1)

    rng = np.random.RandomState(seed=9184)

    pi = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,3))
    ppi = pi.copy()

    qi = np.zeros((N, 1))
    for px in range(N):
        qi[px,0] = (-1.0)**(px+1)
    bias = np.sum(qi[:N:, 0])/N
    qi -= bias


    A.P[:] = pi
    A.Q[:] = qi
    A.GID[:, 0] = np.arange(N)

    A.scatter_data_from(0)
    

    FSD = FreeSpaceDirect()

    def _direct():
        
        _phi_c = FSD(N, ppi, qi)
        return _phi_c

        _phi_direct = 0.0
        # compute phi from image and surrounding 26 cells
        for ix in range(N):
            for jx in range(ix+1, N):
                rij = np.linalg.norm(ppi[jx,:] - ppi[ix,:])
                _phi_direct += qi[ix, 0] * qi[jx, 0] / rij

        return _phi_direct
    
    phi_direct = _direct()


    kmc_fmm = KMCFMM(positions=A.P, charges=A.Q, 
        domain=A.domain, r=R, l=L, boundary_condition='free_space')

    kmc_fmm.initialise()



    for rx in range(200):
    #for rx in range(1):
        
        # using randint breaks seed sync between ranks
        pid = int(rng.uniform(0, 1) * A.npart_local)
        
        #pid = np.where(A.GID.view == 19)

        #if len(pid[0]) > 0:
        #    pid = pid[0][0]

        gid = A.GID[pid, 0]
        pos = rng.uniform(low=-0.5*E, high=0.5*E, size=3)
        
        #print(pid, gid, pos, MPIRANK)

        old_pos = ppi[gid, :].copy()
        ppi[gid, :] = pos

        phi_direct = _direct()

        ppi[gid, :] = old_pos


        prop_energy = kmc_fmm.propose(
            moves=((pid, pos),)
        )
        
        #continue
        assert abs(phi_direct) > 0

        # print(prop_energy[0][0], phi_direct)
        assert (abs(prop_energy[0][0] - phi_direct)/abs(phi_direct) < eps) or (abs(prop_energy[0][0] - phi_direct) < eps)

    kmc_fmm.free()



@pytest.mark.parametrize("R", (3, 4, 5))
def test_kmc_fmm_free_space_accept_1(R):
    """
    Tests proposed moves one by one against direct calculation.
    """

    if R < 4 and MPISIZE > 8:
        return

    eps = 10.**-5
    L = 15

    N = 100
    E = 2*3.1415
    rc = E/4

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()
    A.npart = N

    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)
    A.GID = data.ParticleDat(ncomp=1, dtype=INT64)

    A.crr = data.ScalarArray(ncomp=1)

    rng = np.random.RandomState(seed=9184)

    pi = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,3))
    ppi = pi.copy()

    qi = np.zeros((N, 1))
    for px in range(N):
        qi[px,0] = (-1.0)**(px+1)
    bias = np.sum(qi[:N:, 0])/N
    qi -= bias


    A.P[:] = pi
    A.Q[:] = qi
    A.GID[:, 0] = np.arange(N)
    

    A.scatter_data_from(0)
    

    FSD = FreeSpaceDirect()

    def _direct():
        
        _phi_c = FSD(N, ppi, qi)
        return _phi_c

        _phi_direct = 0.0
        # compute phi from image and surrounding 26 cells
        for ix in range(N):
            for jx in range(ix+1, N):
                rij = np.linalg.norm(ppi[jx,:] - ppi[ix,:])
                _phi_direct += qi[ix, 0] * qi[jx, 0] / rij

        return _phi_direct
    
    phi_direct = _direct()


    kmc_fmm = KMCFMM(positions=A.P, charges=A.Q, 
        domain=A.domain, r=R, l=L, boundary_condition='free_space')
    
    kmc_fmm.initialise()

    for rx in range(200):
        gid = rng.randint(0, N-1)
        pos = rng.uniform(low=-0.5*E, high=0.5*E, size=3)

        pid = np.where(A.GID.view == gid)
        if len(pid[0]) > 0:
            pid = pid[0][0]
            ctrl_rank = True
        else:
            ctrl_rank = False

        old_pos = ppi[gid, :].copy()
        ppi[gid, :] = pos
        phi_direct = _direct()
        ppi[gid, :] = old_pos

        rel = abs(phi_direct)
        rel = 1.0 if rel == 0 else rel
        
        if ctrl_rank:
            
            move = (pid, pos)
            #if A.GID[pid, 0] == 26:
            prop_energy = kmc_fmm.propose(
                moves=(move,)
            )
            
            assert abs(phi_direct) > 0
            assert (abs(prop_energy[0][0] - phi_direct)/rel < eps) or (abs(prop_energy[0][0] - phi_direct) < eps)

        else:
            move = None
        

        kmc_fmm.accept(move)
        
        assert (abs(kmc_fmm.energy - phi_direct)/rel < eps) or (abs(kmc_fmm.energy - phi_direct) < eps)
        
        old_pos = ppi[gid, :].copy()
        ppi[gid, :] = pos       


    kmc_fmm.free()




@pytest.mark.parametrize("R", (3, 4, 5))
def test_kmc_fmm_free_space_accept_2(R):
    """
    Tests proposed moves one by one against direct calculation.
    """

    if R < 4 and MPISIZE > 8:
        return

    eps = 10.**-5
    L = 15

    N = 100
    E = 2*3.1415
    rc = E/4

    max_move = E / 4
    max_move_fixed = 0.9999*2*3.1416 / 4
    assert max_move_fixed <= max_move


    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()
    A.npart = N

    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)
    A.GID = data.ParticleDat(ncomp=1, dtype=INT64)

    A.crr = data.ScalarArray(ncomp=1)

    rng = np.random.RandomState(seed=9184)

    pi = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,3))
    ppi = pi.copy()

    qi = np.zeros((N, 1))
    for px in range(N):
        qi[px,0] = (-1.0)**(px+1)
    bias = np.sum(qi[:N:, 0])/N
    qi -= bias


    A.P[:] = pi
    A.Q[:] = qi
    A.GID[:, 0] = np.arange(N)
    

    A.scatter_data_from(0)
    

    FSD = FreeSpaceDirect()

    def _direct():
        
        _phi_c = FSD(N, ppi, qi)
        return _phi_c

        _phi_direct = 0.0
        # compute phi from image and surrounding 26 cells
        for ix in range(N):
            for jx in range(ix+1, N):
                rij = np.linalg.norm(ppi[jx,:] - ppi[ix,:])
                _phi_direct += qi[ix, 0] * qi[jx, 0] / rij

        return _phi_direct
    
    phi_direct = _direct()


    kmc_fmm = KMCFMM(positions=A.P, charges=A.Q, 
        domain=A.domain, r=R, l=L, boundary_condition='free_space', max_move=max_move)
    
    kmc_fmm.initialise()


    for rx in range(200):

        gid = rng.randint(0, N-1)
        pid = np.where(A.GID.view == gid)

        offset_vector = rng.uniform(low=-1.0, high=1.0, size=3)
        offset_vector /= np.linalg.norm(offset_vector)
        assert abs(np.linalg.norm(offset_vector) - 1.0) < 10.**-15
        offset_size = rng.uniform(low=0.01 * max_move_fixed, high=0.99*max_move_fixed)
        offset_vector *= offset_size
        
        pos = ppi[gid, :].copy() + offset_vector
        

        flush = 0.5
        # flush = 0.49999
        for dimx in range(3):
            pos[dimx] = min(pos[dimx], E*flush)
            pos[dimx] = max(pos[dimx], E*(-flush))


        if len(pid[0]) > 0:
            pid = pid[0][0]
            ctrl_rank = True
        else:
            ctrl_rank = False

        old_pos = ppi[gid, :].copy()
        ppi[gid, :] = pos
        phi_direct = _direct()
        ppi[gid, :] = old_pos

        rel = abs(phi_direct)
        rel = 1.0 if rel == 0 else rel
        
        if ctrl_rank:
            
            move = (pid, pos)

            #print(rx, pid, gid, move)
            prop_energy = kmc_fmm.propose(
                moves=(move,)
            )
            
            err = abs(prop_energy[0][0] - phi_direct)/rel
            #print("\t", err)
            assert abs(phi_direct) > 0
            assert (err < eps) or (abs(prop_energy[0][0] - phi_direct) < eps)

        else:
            move = None
        
        kmc_fmm.accept(move)
        assert (abs(kmc_fmm.energy - phi_direct)/rel < eps) or (abs(kmc_fmm.energy - phi_direct) < eps)
        old_pos = ppi[gid, :].copy()
        ppi[gid, :] = pos       


    kmc_fmm.free()



@pytest.mark.parametrize("R", (3, 4, 5))
@pytest.mark.parametrize("nset", 
    (
        (100, 10),
        (30, 100),
        (2, 400)
    )
)
def test_mpi_free_space_realistic_1(R, nset):
    
    param_boundary = 'free_space'

    if R < 4 and MPISIZE > 8:
        return

    N = nset[0]
    N_steps = nset[1]
    
    L = 12
    
    E = 31.
    rc = E/4
    M = 4
    
    tol_kmc = 10.**-4
    
    max_move = 4.12561


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
    A.GID = data.ParticleDat(ncomp=1, dtype=INT64)

    rng  = np.random.RandomState(seed=817235)
    accept_rng = np.random.RandomState(seed=19512)



    site_max_counts = data.ScalarArray(ncomp=8, dtype=INT64)
    site_max_counts[:] = rng.randint(0, 10, size=8)


    pi = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,3))
    ppi = pi.copy()

    qi = np.zeros((N, 1))
    for px in range(N):
        qi[px,0] = (-1.0)**(px+1)
    bias = np.sum(qi[:N:, 0])/N
    qi -= bias
    

    A.P[:] = pi
    A.Q[:] = qi
    A.GID[:, 0] = np.arange(N)

    A.scatter_data_from(0)

    # create a kmc instance
    kmc_fmmA = KMCFMM(positions=A.P, charges=A.Q, domain=A.domain, r=R, l=L,
        boundary_condition=param_boundary, max_move=max_move)

    kmc_fmmA.initialise()
    

    FSD = FreeSpaceDirect()
    def _direct():
        return FSD(N, ppi, qi)    

    
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
        for px in range(A.npart_local):
            tmp = []
            masks = np.zeros(M)
            masks[:site_max_counts[A.sites[px,0]]:] = 1
            masks = rng.permutation(masks)

            for propx in range(M):
                mask = masks[propx]

                direction = rng.uniform(low=-1.0, high=1.0, size=3)
                direction /= np.linalg.norm(direction)
                direction *= max_move * rng.uniform(0, 1)
                prop_pos = A.P.view[px, :] + direction

                flush = 0.5
                for dimx in range(3):
                    prop_pos[dimx] = min(prop_pos[dimx], E*flush)
                    prop_pos[dimx] = max(prop_pos[dimx], E*(-flush))

                A.prop_masks[px, propx] = mask
                A.prop_positions[px, propx*3:propx*3+3:] = prop_pos
                
                if mask > 0:
                    tmp.append(list(prop_pos))
                    nmov += 1

            if len(tmp) > 0:
                prop.append((px, np.array(tmp)))


        to_test =  kmc_fmmA.propose_with_dats(site_max_counts, A.sites,
            A.prop_positions, A.prop_masks, A.prop_diffs, diff=True)


        found_movs = 0
        for propi, propx in enumerate(prop):
            pid = propx[0]
            movs = propx[1]
            for pmi in range(M):
                if A.prop_masks[pid, pmi] > 0:
                    mov = A.prop_positions[pid, pmi*3:(pmi+1)*3:]
                    gid = A.GID[pid, 0]
                    
                    ppi[gid, :] = mov
                    correct_energy = _direct()
                    ppi[gid, :] = A.P[pid, :]

                    to_test_energy = A.prop_diffs[pid, pmi] + kmc_fmmA.energy
                    
                    rel = 1.0 if abs(correct_energy) < 1 else abs(correct_energy)
                    err = abs(correct_energy - to_test_energy) / rel
                    
                    assert err < tol_kmc

                    found_movs += 1

        assert found_movs == nmov


        # pick random accept
        gid = accept_rng.randint(0, N-1)

        direction = accept_rng.uniform(low=-1.0, high=1.0, size=3)
        direction /= np.linalg.norm(direction)
        direction *= max_move * accept_rng.uniform(0, 1)
        prop_pos = ppi[gid, :] + direction

        flush = 0.5
        for dimx in range(3):
            prop_pos[dimx] = min(prop_pos[dimx], E*flush)
            prop_pos[dimx] = max(prop_pos[dimx], E*(-flush))

        lid = np.where(A.GID.view[:A.npart_local:, 0] == gid)
        # does this rank own the charge
        if len(lid[0]) > 0:
            pid = lid[0][0]
            move = (pid, prop_pos)
            kmc_fmmA.accept(move)
        else:
            kmc_fmmA.accept(None)

        ppi[gid] = prop_pos

        _check_system_energy()



    kmc_fmmA.free()












