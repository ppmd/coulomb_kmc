__author__ = "W.R.Saunders"


import ctypes
import numpy as np
from math import *

REAL = ctypes.c_double
INT64 = ctypes.c_int64

# cuda imports if possible
from ppmd.coulomb.sph_harm import *
from ppmd.lib.build import simple_lib_creator
from coulomb_kmc import common


class LocalExpEval:
    """
    Generates C code to manipulate and evaluate Multipole and Local expansions.
    Also generates the C code for Multipole and Local expansion manipulation
    used in other libraries.

    :arg int L: Number of expansion terms.
    """
    
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

        assign_gen =  'double rhol = 1.0;\n'
        assign_gen += 'double rholcharge = rhol * charge;\n'
        for lx in range(self.L):
            for mx in range(-lx, lx+1):
                assign_gen += 'out[{ind}] += {ylmm} * rholcharge;\n'.format(
                        ind=cube_ind(lx, mx),
                        ylmm=str(sph_gen.get_y_sym(lx, -mx)[0])
                    )
                assign_gen += 'out[IM_OFFSET + {ind}] += {ylmm} * rholcharge;\n'.format(
                        ind=cube_ind(lx, mx),
                        ylmm=str(sph_gen.get_y_sym(lx, -mx)[1])
                    )
            assign_gen += 'rhol *= radius;\n'
            assign_gen += 'rholcharge = rhol * charge;\n'

        src = """
        #define IM_OFFSET ({IM_OFFSET})

        {DECLARE} int multipole_exp(
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
        """
        header = str(sph_gen.header)
        
        src_lib = src.format(
            SPH_GEN=str(sph_gen.module),
            ASSIGN_GEN=str(assign_gen),
            IM_OFFSET=(self.L**2),
            DECLARE=r'static inline'
        )

        src = src.format(
            SPH_GEN=str(sph_gen.module),
            ASSIGN_GEN=str(assign_gen),
            IM_OFFSET=(self.L**2),
            DECLARE=r'extern "C"'
        )

        self.create_multipole_header = header
        self.create_multipole_src = src_lib

        self._multipole_lib = simple_lib_creator(header_code=header, src_code=src)['multipole_exp']


        # --- lib to create vector to dot product with local expansions --- 

        assign_gen =  'double rhol = 1.0;\n'
        assign_gen += 'double rholcharge = rhol * charge;\n'

        for lx in range(self.L):
            for mx in range(-lx, lx+1):
                assign_gen += 'out[{ind}] += {ylmm} * rholcharge;\n'.format(
                        ind=cube_ind(lx, mx),
                        ylmm=str(sph_gen.get_y_sym(lx, mx)[0])
                    )
                assign_gen += 'out[IM_OFFSET + {ind}] += (-1.0) * {ylmm} * rholcharge;\n'.format(
                        ind=cube_ind(lx, mx),
                        ylmm=str(sph_gen.get_y_sym(lx, mx)[1])
                    )
            assign_gen += 'rhol *= radius;\n'
            assign_gen += 'rholcharge = rhol * charge;\n'

        src = """
        #define IM_OFFSET ({IM_OFFSET})

        {DECLARE} int local_dot_vec(
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
        """
        header = str(sph_gen.header)
        
        src_lib = src.format(
            SPH_GEN=str(sph_gen.module),
            ASSIGN_GEN=str(assign_gen),
            IM_OFFSET=(self.L**2),
            DECLARE=r'static inline'
        )

        src = src.format(
            SPH_GEN=str(sph_gen.module),
            ASSIGN_GEN=str(assign_gen),
            IM_OFFSET=(self.L**2),
            DECLARE=r'extern "C"'
        )

        self.create_dot_vec_header = header
        self.create_dot_vec_src = src_lib

        self._dot_vec_lib = simple_lib_creator(header_code=header, src_code=src)['local_dot_vec']


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

        {DECLARE} int local_eval(
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
        """
        
        src_lib = src.format(
            SPH_GEN=str(sph_gen.module),
            ASSIGN_GEN=str(assign_gen),
            IM_OFFSET=(self.L**2),
            DECLARE=r'static inline'
        )

        src = src.format(
            SPH_GEN=str(sph_gen.module),
            ASSIGN_GEN=str(assign_gen),
            IM_OFFSET=(self.L**2),
            DECLARE=r'extern "C"'
        )
        header = str(sph_gen.header)


        self.create_local_eval_header = header
        self.create_local_eval_src = src_lib

        self._local_eval_lib = simple_lib_creator(header_code=header, src_code=src)['local_eval']

        # lib to create local expansions
        
        tflops = common.new_flop_dict()
        tflops = common.add_flop_dict(tflops, sph_gen.flops)

        assign_gen = 'const double iradius = 1.0/radius;\n'
        assign_gen += 'double rhol = iradius;\n'
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
                tflops['+'] += 2
                tflops['*'] += 4
            assign_gen += 'rhol *= iradius;\n'
            tflops['*'] += 1
        
        self.flop_count_create_local_exp = tflops

        src = """
        #define IM_OFFSET ({IM_OFFSET})

        extern "C" int create_local_exp(
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


        self.create_local_exp_header = header
        self.create_local_exp_src = """
        #define IM_OFFSET ({IM_OFFSET})

        static inline void inline_local_exp(
            const double charge,
            const double radius,
            const double theta,
            const double phi,
            double * RESTRICT out
        ){{
            {SPH_GEN}
            {ASSIGN_GEN}
            return;
        }}
        """.format(
            SPH_GEN=str(sph_gen.module),
            ASSIGN_GEN=str(assign_gen),
            IM_OFFSET=(self.L**2),
        )

        self._local_create_lib = simple_lib_creator(header_code=header, src_code=src)['create_local_exp']


        # --- lib to create vector to dot product and mutlipole expansions --- 

        assign_gen =  'double rhol = 1.0;\n'
        assign_gen += 'double rholcharge = rhol * charge;\n'
        flops = {'+': 0, '-': 0, '*': 0, '/': 0}

        for lx in range(self.L):
            for mx in range(-lx, lx+1):

                assign_gen += 'out_mul[{ind}] += {ylmm} * rholcharge;\n'.format(
                        ind=cube_ind(lx, mx),
                        ylmm=str(sph_gen.get_y_sym(lx, -mx)[0])
                    )
                assign_gen += 'out_mul[IM_OFFSET + {ind}] += {ylmm} * rholcharge;\n'.format(
                        ind=cube_ind(lx, mx),
                        ylmm=str(sph_gen.get_y_sym(lx, -mx)[1])
                    )
                assign_gen += 'out_vec[{ind}] += {ylmm} * rholcharge;\n'.format(
                        ind=cube_ind(lx, mx),
                        ylmm=str(sph_gen.get_y_sym(lx, mx)[0])
                    )
                assign_gen += 'out_vec[IM_OFFSET + {ind}] += (-1.0) * {ylmm} * rholcharge;\n'.format(
                        ind=cube_ind(lx, mx),
                        ylmm=str(sph_gen.get_y_sym(lx, mx)[1])
                    )

                flops['+'] += 4
                flops['*'] += 5

            assign_gen += 'rhol *= radius;\n'
            assign_gen += 'rholcharge = rhol * charge;\n'
            flops['*'] += 2

        flops['+'] += sph_gen.flops['*']
        flops['-'] += sph_gen.flops['*']
        flops['*'] += sph_gen.flops['*']
        flops['/'] += sph_gen.flops['*']

        src = """
        #define IM_OFFSET ({IM_OFFSET})

        {DECLARE} int local_dot_vec_multipole(
            const double charge,
            const double radius,
            const double theta,
            const double phi,
            double * RESTRICT out_vec,
            double * RESTRICT out_mul
        ){{
            {SPH_GEN}
            {ASSIGN_GEN}
            return 0;
        }}
        """
        header = str(sph_gen.header)
        
        src_lib = src.format(
            SPH_GEN=str(sph_gen.module),
            ASSIGN_GEN=str(assign_gen),
            IM_OFFSET=(self.L**2),
            DECLARE='static inline'
        )

        src = src.format(
            SPH_GEN=str(sph_gen.module),
            ASSIGN_GEN=str(assign_gen),
            IM_OFFSET=(self.L**2),
            DECLARE=r'extern "C"'
        )

        self.create_dot_vec_multipole_header = header
        self.create_dot_vec_multipole_src = src_lib
        self.create_dot_vec_multipole_flops = flops

        self._dot_vec_multipole_lib = simple_lib_creator(header_code=header, src_code=src)['local_dot_vec_multipole']
        

    def dot_vec_multipole(self, sph, charge, arr_vec, arr_mul):
        """
        For a charge at the point sph computes the coefficients at the origin
        and appends them onto arr that can be used in a dot product to compute
        the energy.
        """
        assert arr_vec.dtype == REAL
        assert arr_mul.dtype == REAL
        self._dot_vec_multipole_lib(
            REAL(charge),
            REAL(sph[0]),
            REAL(sph[1]),
            REAL(sph[2]),
            arr_vec.ctypes.get_as_parameter(),
            arr_mul.ctypes.get_as_parameter()
        )    


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

    def local_exp(self, sph, charge, arr):
        """
        For a charge at the point sph computes the local expansion at the origin
        and appends it onto arr.
        """

        assert arr.dtype == REAL
        self._local_create_lib(
            REAL(charge),
            REAL(sph[0]),
            REAL(sph[1]),
            REAL(sph[2]),
            arr.ctypes.get_as_parameter()
        )

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


    def dot_vec(self, sph, charge, arr):
        """
        For a charge at the point sph computes the coefficients at the origin
        and appends them onto arr that can be used in a dot product to compute
        the energy.
        """
        assert arr.dtype == REAL
        self._dot_vec_lib(
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
