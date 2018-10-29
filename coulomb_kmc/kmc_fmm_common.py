


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
from ppmd.coulomb.sph_harm import *
from ppmd.lib.build import simple_lib_creator

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
        
        self.sph_gen = SphGen(L-1)
        self._multipole_lib = None
        self._generate_host_libs()

    def _generate_host_libs(self):

        sph_gen = self.sph_gen

        def cube_ind(L, M):
            return ((L) * ( (L) + 1 ) + (M) )

        assign_gen = 'double rhol = 1.0;\n'
        for lx in range(self.L):
            for mx in range(-lx, lx+1):
                assign_gen += 'out[{ind}] += {ylmm} * rhol * charge;\n'.format(
                        ind=cube_ind(lx, mx),
                        ylmm=str(sph_gen.get_y_sym(lx, -mx)[0])
                    )
                assign_gen += 'out[IM_OFFSET + {ind}] += {ylmm} * rhol * charge;\n'.format(
                        ind=cube_ind(lx, mx),
                        ylmm=str(sph_gen.get_y_sym(lx, -mx)[1])
                    )
            assign_gen += 'rhol *= radius;\n'

        src = """
        #define IM_OFFSET ({IM_OFFSET})

        extern "C" int multipole_exp(
            const double charge,
            const double radius,
            const double theta,
            const double phi,
            double * RESTRICT out
        ){{
            {SPH_GEN}
            {ASSIGN_GEN}
            return 0;
        }}
        """.format(
            SPH_GEN=str(sph_gen.module),
            ASSIGN_GEN=str(assign_gen),
            IM_OFFSET=(self.L**2),
        )
        header = str(sph_gen.header)

        self._multipole_lib = simple_lib_creator(header_code=header, src_code=src)['multipole_exp']


        # --- lib to evaluate local expansions --- 

        assign_gen = ''
        for lx in range(self.L):
            for mx in range(-lx, lx+1):
                reL = SphSymbol('moments[{ind}]'.format(ind=cube_ind(lx, mx)))
                imL = SphSymbol('moments[IM_OFFSET + {ind}]'.format(ind=cube_ind(lx, mx)))
                reY, imY = sph_gen.get_y_sym(lx, mx)
                phi_sym = cmplx_mul(reL, imL, reY, imY)[0]
                assign_gen += 'tmp_energy += rhol * ({phi_sym});\n'.format(phi_sym=str(phi_sym))

            assign_gen += 'rhol *= radius;\n'

        src = """
        #define IM_OFFSET ({IM_OFFSET})

        extern "C" int local_eval(
            const double radius,
            const double theta,
            const double phi,
            const double * RESTRICT moments,
            double * RESTRICT out
        ){{
            {SPH_GEN}
            double rhol = 1.0;
            double tmp_energy = 0.0;
            {ASSIGN_GEN}

            out[0] = tmp_energy;
            return 0;
        }}
        """.format(
            SPH_GEN=str(sph_gen.module),
            ASSIGN_GEN=str(assign_gen),
            IM_OFFSET=(self.L**2),
        )
        header = str(sph_gen.header)

        self._local_eval_lib = simple_lib_creator(header_code=header, src_code=src)['local_eval']

    def compute_phi_local(self, moments, disp_sph):
        assert moments.dtype == REAL
        _out = REAL(0.0)
        self._local_eval_lib(
            REAL(disp_sph[0]),
            REAL(disp_sph[1]),
            REAL(disp_sph[2]),
            moments.ctypes.get_as_parameter(),
            ctypes.byref(_out)
        )
        return _out.value, None


    def py_compute_phi_local(self, moments, disp_sph):
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

        assert arr.dtype == REAL
        self._multipole_lib(
            REAL(charge),
            REAL(sph[0]),
            REAL(sph[1]),
            REAL(sph[2]),
            arr.ctypes.get_as_parameter()
        )

    def py_multipole_exp(self, sph, charge, arr):
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


