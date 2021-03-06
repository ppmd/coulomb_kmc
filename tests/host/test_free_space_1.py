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
from coulomb_kmc.common import spherical
from coulomb_kmc.kmc_expansion_tools import LocalExpEval

_PROFILE = common.PROFILE


def red(*input):
    try:
        from termcolor import colored
        return colored(*input, color='red')
    except Exception as e: return input
def green(*input):
    try:
        from termcolor import colored
        return colored(*input, color='green')
    except Exception as e: return input
def yellow(*input):
    try:
        from termcolor import colored
        return colored(*input, color='yellow')
    except Exception as e: return input

def red_tol(val, tol):
    if abs(val) > tol:
        return red(str(val))
    else:
        return green(str(val))


def get_fmm_cell(s, ix, R):
    cc = s._fmm_cell[ix][0]
    sl = 2 ** (R - 1)
    cx = cc % sl
    cycz = (cc - cx) // sl
    cy = cycz % sl
    cz = (cycz - cy) // sl
    return cx, cy, cz

def get_cell_disp(s, ix, R):
    sl = 2 ** (R - 1)
    csl = [s.domain.extent[0] / sl,
           s.domain.extent[1] / sl,
           s.domain.extent[2] / sl]
    
    es = [s.domain.extent[0] * -0.5,
          s.domain.extent[1] * -0.5,
          s.domain.extent[2] * -0.5]
    

    cc = get_fmm_cell(s, ix, R)

    ec = [esx + 0.5 * cx + ccx * cx for esx, cx, ccx in zip(es, csl, cc)]
    px = (s.P[ix, 0], s.P[ix, 1], s.P[ix, 2])
    
    disp = (px[0] - ec[0], px[1] - ec[1], px[2] - ec[2])
    sph = spherical(disp)
    
    return sph


def get_local_expansion(fmm, cell):
    ls = fmm.tree[fmm.R-1].local_grid_cube_size
    lo = fmm.tree[fmm.R-1].local_grid_offset
    lor = list(lo)
    lor.reverse()
    lc = [cx - lx for cx, lx in zip(cell, lor)]
    return fmm.tree_plain[fmm.R-1][lc[2], lc[1], lc[0], :]


def compute_phi_local(llimit, moments, disp_sph):

    phi_sph_re = 0.
    phi_sph_im = 0.
    def re_lm(l,m): return (l**2) + l + m
    def im_lm(l,m): return (l**2) + l +  m + llimit**2

    for lx in range(llimit):
        mrange = list(range(lx, -1, -1)) + list(range(1, lx+1))
        mrange2 = list(range(-1*lx, 1)) + list(range(1, lx+1))
        scipy_p = lpmv(mrange, lx, np.cos(disp_sph[1]))

        #print('lx', lx, '-------------')

        for mxi, mx in enumerate(mrange2):

            re_exp = np.cos(mx*disp_sph[2])
            im_exp = np.sin(mx*disp_sph[2])

            #print('mx', mx, im_exp)

            val = math.sqrt(math.factorial(
                lx - abs(mx))/math.factorial(lx + abs(mx)))
            val *= scipy_p[mxi]

            irad = disp_sph[0] ** (lx)

            scipy_real = re_exp * val * irad
            scipy_imag = im_exp * val * irad

            ppmd_mom_re = moments[re_lm(lx, mx)]
            ppmd_mom_im = moments[im_lm(lx, mx)]

            phi_sph_re += scipy_real*ppmd_mom_re - scipy_imag*ppmd_mom_im
            phi_sph_im += scipy_real*ppmd_mom_im + ppmd_mom_re*scipy_imag

    return phi_sph_re, phi_sph_im

def charge_indirect_energy(s, ix, fmm):
    cell = get_fmm_cell(s, ix, fmm.R)
    lexp = get_local_expansion(fmm, cell)
    disp = get_cell_disp(s, ix, fmm.R)
    return s.Q[ix,0] * compute_phi_local(fmm.L, lexp, disp)[0]



@pytest.mark.skipif('MPISIZE > 1')
@pytest.mark.parametrize("R", (3, 4, 5))
def test_kmc_fmm_free_space_1(R):
    """
    Tests proposed moves one by one against direct calculation.
    """

    eps = 10.**-5
    L = 12

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
    
    A.PP[:] = A.P[:]

    def _direct():
        _phi_direct = 0.0
        # compute phi from image and surrounding 26 cells
        for ix in range(N):
            phi_part = 0.0
            for jx in range(ix+1, N):
                rij = np.linalg.norm(A.PP[jx,:] - A.PP[ix,:])
                _phi_direct += A.Q[ix, 0] * A.Q[jx, 0] /rij
        return _phi_direct
    
    phi_direct = _direct()


    kmc_fmm = KMCFMM(positions=A.P, charges=A.Q, 
        domain=A.domain, r=R, l=L, boundary_condition='free_space')
    
    kmc_fmm.initialise()


    for rx in range(2*N):
        pid = rng.randint(0, N-1)

        pos = rng.uniform(low=-0.5*E, high=0.5*E, size=3)
        
        A.PP[:] = A.P[:]
        A.PP[pid, :] = pos

        phi_direct = _direct()

        prop_energy = kmc_fmm.test_propose(
            moves=((pid, pos),)
        )
        
        assert abs(phi_direct) > 0

        # print(prop_energy[0][0], phi_direct)
        assert abs(prop_energy[0][0] - phi_direct)/abs(phi_direct) < eps

    kmc_fmm.free()
    

@pytest.mark.skipif('MPISIZE > 1')
def test_kmc_fmm_free_space_2():
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
    
    A.PP[:] = A.P[:]

    def _direct():
        _phi_direct = 0.0
        # compute phi from image and surrounding 26 cells
        for ix in range(N):
            phi_part = 0.0
            for jx in range(ix+1, N):
                rij = np.linalg.norm(A.PP[jx,:] - A.PP[ix,:])
                _phi_direct += A.Q[ix, 0] * A.Q[jx, 0] /rij
        return _phi_direct
    


    # create a kmc instance
    kmc_fmm = KMCFMM(positions=A.P, charges=A.Q, 
        domain=A.domain, r=R, l=L, boundary_condition='free_space')
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
    prop_energy = kmc_fmm.test_propose(moves=prop)
    
    # test agains the direct calculation
    for rxi, rx in enumerate(prop):
        pid = rx[0]
        for movi, mov in enumerate(rx[1]):
            
            A.PP[pid, :] = mov

            phi_direct = _direct()
            
            A.PP[pid, :] = A.P[pid, :]
        
            assert abs(phi_direct) > 0
            
            fmm_phi = prop_energy[rxi][movi]

            assert abs(fmm_phi - phi_direct)/abs(phi_direct) < eps

    kmc_fmm.free()


@pytest.mark.skipif('MPISIZE > 1')
@pytest.mark.parametrize("R", (3, 4, 5))
def test_kmc_fmm_free_space_3(R):
    """
    Passes all proposed moves to kmc at once, then checks all outputs
    """
    
    eps = 10.**-5
    L = 10

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
        domain=A.domain, r=R, l=L, boundary_condition='free_space')
    kmc_fmm.initialise()
    
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
    
    # get the energy of the proposed moves

    #print("\nN :", N, "\nNMOVES:", nmov)
    
    import cProfile
    pr = cProfile.Profile()
    pr.enable()
    t0 = time.time()
    prop_energy_c  = kmc_fmm.propose(moves=prop)
    t1 = time.time()
    pr.disable()
    #pr.dump_stats('/tmp/propose.prof')
    #print("C :", t1 - t0)
    #common.print_profile()

    prop_energy_py = kmc_fmm.test_propose(moves=prop, use_python=True)
    t2 = time.time()

    #print("PY:", t2 - t1)
    #common.print_profile()

    # test agains the direct calculation
    for rxi, rx in enumerate(prop):
        pid = rx[0]
        for movi, mov in enumerate(rx[1]):
            
            fmm_phi_py = prop_energy_py[rxi][movi]
            fmm_phi_c = prop_energy_c[rxi][movi]
            
            assert abs(fmm_phi_py - fmm_phi_c)/abs(fmm_phi_py) < eps

    #from coulomb_kmc.common import print_profile
    #print("\n")
    #print_profile()

    kmc_fmm.free()

@pytest.mark.skipif('MPISIZE > 1')
def test_kmc_fmm_free_space_accept_1():
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
        domain=A.domain, r=R, l=L, boundary_condition='free_space')
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
    
    kmc_fmm.free()


@pytest.mark.skipif('MPISIZE > 1')
def test_kmc_fmm_free_space_accept_1_5():

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
        domain=A.domain, r=R, l=L, boundary_condition='free_space')
    kmc_fmmA.initialise()
    
    kmc_fmmB = KMCFMM(positions=B.P, charges=B.Q, 
        domain=B.domain, r=R, l=L, boundary_condition='free_space')
    kmc_fmmB.initialise() 
    
    # print("\n arggggg")

    for stepx in range(3):
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

            assert err<10.**-14

    kmc_fmmA.free()
    kmc_fmmB.free()



@pytest.mark.skipif('MPISIZE > 1')
def test_kmc_fmm_free_space_accept_2():

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
        domain=A.domain, r=R, l=L, boundary_condition='free_space')
    kmc_fmmA.initialise()
    
    kmc_fmmB = KMCFMM(positions=B.P, charges=B.Q, 
        domain=B.domain, r=R, l=L, boundary_condition='free_space')
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
        
        #print("====")
        #print(A.P[prop[0][0], :])
        #print("~~~")
        #print(prop)
        #print("====")

        prop_energyA  = kmc_fmmA.propose(moves=prop)
        prop_energyB  = kmc_fmmB.propose(moves=prop)
        
    
        for pa, pb in zip(prop_energyA, prop_energyB):
            for pai, pbi in zip(pa, pb):
                rel = abs(pai) if abs(pai) > 1.0 else 1.0
                err = abs(pai - pbi) / rel
                #print(err)
                assert err < 10.**-6


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
        #print("---", err)
        assert err < 10.**-6

    kmc_fmmA.free()
    kmc_fmmB.free()


























