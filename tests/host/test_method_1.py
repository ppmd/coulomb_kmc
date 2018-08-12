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


@pytest.mark.skipif("MPISIZE>1")
def test_method_1():

    R = 3
    eps = 10.**-6
    L = 10
    free_space = True

    N = 4
    E = 4.
    rc = E/4

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    ASYNC = False
    DIRECT = True if MPISIZE == 1 else False

    DIRECT= True
    EWALD = True

    fmm = PyFMM(domain=A.domain, r=R, l=L, free_space=free_space)

    A.npart = N

    rng = np.random.RandomState(seed=1234)

    A.P = data.PositionDat(ncomp=3)
    A.F = data.ParticleDat(ncomp=3)
    A.FE = data.ParticleDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)

    A.crr = data.ScalarArray(ncomp=1)
    A.cri = data.ScalarArray(ncomp=1)
    A.crs = data.ScalarArray(ncomp=1)


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

    elif N == 2:

        A.P[0,:] = ( 1.51,  1.51, 0.0)
        A.P[1,:] = (-1.49,  1.51, 0.0)

        A.Q[0,0] = 1.
        A.Q[1,0] = 1.



    A.scatter_data_from(0)

    t0 = time.time()
    phi_py = fmm(A.P, A.Q, forces=A.F, async=ASYNC)
    t1 = time.time()



    direct_forces = np.zeros((N, 3))
    
    def _direct():
        #print("WARNING 0-th PARTICLE ONLY")
        _phi_direct = 0.0

        # compute phi from image and surrounding 26 cells

        for ix in range(N):

            phi_part = 0.0
            for jx in range(ix+1, N):
                rij = np.linalg.norm(A.P[jx,:] - A.P[ix,:])
                _phi_direct += A.Q[ix, 0] * A.Q[jx, 0] /rij

        return _phi_direct
    
    phi_direct = _direct()


    def print_diff():

        local_err = abs(phi_py - phi_direct)
        if local_err > eps: serr = red(local_err)
        else: serr = green(local_err)

        print("\n")
        print("ENERGY DIRECT:\t{:.20f}".format(phi_direct))
        print("ENERGY FMM:\t", phi_py)
        print("ERR:\t\t", serr)

    if MPIRANK == 0 and DEBUG:
        print_diff()
    
    print(60 * '-')

    
    ui = [charge_indirect_energy(A, px, fmm) for px in range(N)]
    print("ui", ui)
    print(sum(ui) * 0.5)
    
    u1 = charge_indirect_energy(A, 0, fmm)
    A.P[0,:] = ( 1.6,  1.7, 0.0)
    
    u2 = charge_indirect_energy(A, 0, fmm)
    
    phi_py = phi_py - u1 + u2 

    phi_direct = _direct()
    print_diff()
    
    
    phi_py = fmm(A.P, A.Q, forces=A.F, async=ASYNC)


    print_diff()



def test_kmc_fmm_1():
    
    L = 10
    R = 3

    N = 4
    E = 4.
    rc = E/4

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    fmm = KMCFMM(domain=A.domain, r=R, l=L, boundary_condition='pbc')










