from __future__ import print_function, division

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"

import pytest, ctypes, math
from mpi4py import MPI
import numpy as np

np.set_printoptions(linewidth=200)

import itertools
def get_res_file_path(filename):
    return os.path.join(os.path.join(os.path.dirname(__file__), '../../res'), filename)


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

def spherical(xyz):
    if type(xyz) is tuple:
        sph = np.zeros(3)
        xy = xyz[0]**2 + xyz[1]**2
        # r
        sph[0] = np.sqrt(xy + xyz[2]**2)
        # polar angle
        sph[1] = np.arctan2(np.sqrt(xy), xyz[2])
        # longitude angle
        sph[2] = np.arctan2(xyz[1], xyz[0])

    else:
        sph = np.zeros(xyz.shape)
        xy = xyz[:,0]**2 + xyz[:,1]**2
        # r
        sph[:,0] = np.sqrt(xy + xyz[:,2]**2)
        # polar angle
        sph[:,1] = np.arctan2(np.sqrt(xy), xyz[:,2])
        # longitude angle
        sph[:,2] = np.arctan2(xyz[:,1], xyz[:,0])

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




def test_kmc_fmm_free_space_1():
    """
    Tests proposed moves one by one against direct calculation.
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


def test_kmc_fmm_free_space_2():
    """
    Passes all proposed moves to kmc at once, then checks all outputs
    """
    
    eps = 10.**-5
    L = 12
    R = 3

    N = 20
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
























