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
from itertools import product
MPISIZE = MPI.COMM_WORLD.Get_size()
MPIRANK = MPI.COMM_WORLD.Get_rank()
MPIBARRIER = MPI.COMM_WORLD.Barrier
DEBUG = True
SHARED_MEMORY = 'omp'


from coulomb_kmc import *
from coulomb_kmc.common import BCType
from coulomb_kmc.kmc_fmm_common import *

INT64 = kmc_octal.INT64
REAL = kmc_octal.REAL





def test_c_sph_harm_1():
    rng = np.random.RandomState(9476213)
    N = 20
    L = 20
    ncomp = (L**2)*2

    offsets = np.zeros(N, dtype=INT64)
    positions = np.array(rng.uniform(low=0., high=1.0, size=(N, 3)), dtype=REAL)
    charges = np.array(rng.uniform(size=N), dtype=REAL)
    moments = np.array(rng.uniform(low=0.0, high=1.0, size=ncomp), dtype=REAL)
    centres = np.array((0.5, 0.5, 0.5), dtype=REAL)
    energy = np.zeros(N, dtype=REAL)
    
    lib = kmc_octal.LocalCellExpansions._init_host_kernels(L)
    
    def cptr(arr):
        return arr.ctypes.get_as_parameter()

    lib(
        INT64(N),
        cptr(offsets),
        cptr(centres),
        cptr(positions),
        cptr(charges),
        cptr(moments),
        cptr(energy)
    )
    
    lee = LocalExpEval(L)

    for ix in range(N):
        pos = tuple(positions[ix, :] - centres)
        sph_coord = spherical(pos)
        e_py = lee.compute_phi_local(moments, sph_coord)[0] * charges[ix]

        err = abs(e_py - energy[ix])
        
        assert err < 10.**-14


def test_c_sph_harm_2():
    rng = np.random.RandomState(9476213)
    N = 2000
    L = 12
    ncomp = (L**2)*2

    offsets = np.zeros(N, dtype=INT64)
    positions = np.array(rng.uniform(low=0., high=1.0, size=(N, 3)), dtype=REAL)
    charges = np.array(rng.uniform(size=N), dtype=REAL)
    moments = np.array(rng.uniform(low=0.0, high=1.0, size=ncomp), dtype=REAL)
    centres = np.array((0.5, 0.5, 0.5), dtype=REAL)
    energy = np.zeros(N, dtype=REAL)
    
    lib = kmc_octal.LocalCellExpansions._init_host_kernels(L)
    
    lee = LocalExpEval(L)

    def cptr(arr):
        return arr.ctypes.get_as_parameter()
    
    t0 = time.time()
    lib(
        INT64(N),
        cptr(offsets),
        cptr(centres),
        cptr(positions),
        cptr(charges),
        cptr(moments),
        cptr(energy)
    )
    t1 = time.time()

    # print("C", t1 - t0)

    for ix in range(N):
        pos = tuple(positions[ix, :] - centres)
        sph_coord = spherical(pos)
        e_py = lee.compute_phi_local(moments, sph_coord)[0] * charges[ix]

        err = abs(e_py - energy[ix])
        
        assert err < 10.**-14


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


def test_c_local_dot_eval():
    L = 2
    lee = LocalExpEval(L)
    rng = np.random.RandomState(9476213)

    ncomp = (L**2)*2

    for tx in range(50):
        L_exp = np.array(rng.uniform(size=ncomp), dtype=REAL)
        L_coe = np.zeros_like(L_exp)
        
        pos = (tuple(rng.uniform(size=3)))
        sph_pos = spherical(pos)
        lee.dot_vec(sph_pos, 1, L_coe)

        eng_c = np.dot(L_coe, L_exp)

        eng_p, _ = compute_phi_local(L, L_exp, sph_pos)

        err = abs(eng_c - eng_p) / abs(eng_c)
        assert err < 10.**-14








