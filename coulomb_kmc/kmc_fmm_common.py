


import ctypes
import numpy as np
from math import *
import scipy
from scipy.special import lpmv

REAL = ctypes.c_double
INT64 = ctypes.c_int64

from coulomb_kmc.common import BCType, PROFILE

# cuda imports if possible
import ppmd
import ppmd.cuda

if ppmd.cuda.CUDA_IMPORT:
    cudadrv = ppmd.cuda.cuda_runtime.cudadrv
    # the device should be initialised already by ppmd
    from pycuda.compiler import SourceModule
    import pycuda.gpuarray as gpuarray


class LocalExpEval(object):
    
    def __init__(self, L):
        self.L = L
        self._hmatrix_py = np.zeros((2*self.L, 2*self.L))
        def Hfoo(nx, mx): return sqrt(float(factorial(nx - abs(mx)))/factorial(nx + abs(mx)))
        for nx in range(self.L):
            for mx in range(-nx, nx+1):
                self._hmatrix_py[nx, mx] = Hfoo(nx, mx)

    def compute_phi_local(self, moments, disp_sph):
        """
        Computes the field at the podint disp_sph given by the local expansion 
        in moments
        """

        llimit = self.L
    
        phi_sph_re = 0.
        phi_sph_im = 0.
        def re_lm(l,m): return (l**2) + l + m
        def im_lm(l,m): return (l**2) + l +  m + llimit**2

        cosv = np.zeros(3 * llimit)
        sinv = np.zeros(3 * llimit)
        for mx in range(-llimit, llimit+1):
            cosv[mx] = cos(mx * disp_sph[2])
            sinv[mx] = sin(mx * disp_sph[2])

        for lx in range(llimit):
            scipy_p = lpmv(range(lx+1), lx, np.cos(disp_sph[1]))
            irad = disp_sph[0] ** (lx)
            for mx in range(-lx, lx+1):

                val = self._hmatrix_py[lx, mx] * scipy_p[abs(mx)]

                scipy_real = cosv[mx] * val * irad
                scipy_imag = sinv[mx] * val * irad

                ppmd_mom_re = moments[re_lm(lx, mx)]
                ppmd_mom_im = moments[im_lm(lx, mx)]

                phi_sph_re += scipy_real*ppmd_mom_re - scipy_imag*ppmd_mom_im
                phi_sph_im += scipy_real*ppmd_mom_im + ppmd_mom_re*scipy_imag

        return phi_sph_re, phi_sph_im

    def multipole_exp(self, sph, charge, arr):
        """
        For a charge at the point sph computes the multipole moments at the origin
        and appends them onto arr.
        """

        llimit = self.L
        def re_lm(l,m): return (l**2) + l + m
        def im_lm(l,m): return (l**2) + l +  m + llimit**2
        
        cosv = np.zeros(3 * llimit)
        sinv = np.zeros(3 * llimit)
        for mx in range(-llimit, llimit+1):
            cosv[mx] = cos(-1.0 * mx * sph[2])
            sinv[mx] = sin(-1.0 * mx * sph[2])

        for lx in range(self.L):
            scipy_p = lpmv(range(lx+1), lx, cos(sph[1]))
            radn = sph[0] ** lx
            for mx in range(-lx, lx+1):
                coeff = charge * radn * self._hmatrix_py[lx, mx] * scipy_p[abs(mx)] 
                arr[re_lm(lx, mx)] += cosv[mx] * coeff
                arr[im_lm(lx, mx)] += sinv[mx] * coeff


class LocalOctalBase:

    def _profile_inc(self, key, inc):
        key = self.__class__.__name__ + ':' + key
        if key not in PROFILE.keys():
            PROFILE[key] = inc
        else:
            PROFILE[key] += inc

    def _get_fmm_cell(self, ix, cell_map, slow_to_fast=False):
        # produces xyz tuple by default
        R = self.fmm.R
        cc = cell_map[ix][0]
        sl = 2 ** (R - 1)
        cx = cc % sl
        cycz = (cc - cx) // sl
        cy = cycz % sl
        cz = (cycz - cy) // sl
        
        els = self.entry_local_size
        elo = self.entry_local_offset

        assert cz >= elo[0] and cz < elo[0] + els[0]
        assert cy >= elo[1] and cy < elo[1] + els[1]
        assert cx >= elo[2] and cx < elo[2] + els[2]

        if not slow_to_fast:
            return cx, cy, cz
        else:
            return cz, cy, cx

    def _global_cell_xyz(self, tcx):
        """get global cell index from xyz tuple"""
        gcs = self.global_cell_size
        return tcx[0] + gcs[0] * ( tcx[1] + gcs[1] * tcx[2] )


