

import ctypes
import ppmd
from ppmd.lib import build

REAL = ctypes.c_double
INT64 = ctypes.c_int64

from itertools import product

from coulomb_kmc.common import spherical
from ppmd.coulomb.fmm_pbc import LongRangeMTL
from coulomb_kmc.kmc_expansion_tools import LocalExpEval

import numpy as np

class FreeSpaceDirect:
    def __init__(self):
        
        
        header = r"""
        #include <math.h>
        #define INT64 int64_t
        #define REAL double
        """

        src = r"""
        
        extern "C" int free_space_direct(
            const INT64 N,
            const REAL * RESTRICT P,
            const REAL * RESTRICT Q,
            REAL * RESTRICT phi
        ){{

            REAL tmp_phi = 0.0;

            #pragma omp parallel for reduction(+:tmp_phi)
            for(INT64 ix=0 ; ix<N ; ix++){{
                REAL tmp_inner_phi = 0.0;
                
                const REAL iq = Q[ix];
                const REAL ip0 = P[3*ix + 0];
                const REAL ip1 = P[3*ix + 1];
                const REAL ip2 = P[3*ix + 2];

                
                #pragma omp simd reduction(+:tmp_inner_phi)
                for(INT64 jx=(ix+1) ; jx<N ; jx++){{
                    
                    const REAL jq = Q[jx];
                    const REAL jp0 = P[3*jx + 0];
                    const REAL jp1 = P[3*jx + 1];
                    const REAL jp2 = P[3*jx + 2];

                    const REAL d0 = ip0 - jp0;
                    const REAL d1 = ip1 - jp1;
                    const REAL d2 = ip2 - jp2;
                    
                    const REAL r2 = d0*d0 + d1*d1 + d2*d2;
                    const REAL r = sqrt(r2);

                    tmp_inner_phi += iq * jq / r;

                }}
                
                tmp_phi += tmp_inner_phi;

            }}
           
            phi[0] = tmp_phi;
            return 0;
        }}
        """.format()


        self._lib = build.simple_lib_creator(header_code=header, src_code=src, name="kmc_fmm_free_space_direct")['free_space_direct']
    

    def __call__(self, N, P, Q):

        phi = ctypes.c_double(0)

        self._lib(
            INT64(N),
            P.ctypes.get_as_parameter(),
            Q.ctypes.get_as_parameter(),
            ctypes.byref(phi)
        )
        
        return phi.value


class NearestDirect:
    def __init__(self, E):

        ox_range = tuple(range(-1, 2))

        inner = ''

        for oxi, ox in enumerate(product(ox_range, ox_range, ox_range)):
                if ox[0] != 0 or ox[1] != 0 or ox[2] != 0:
                    inner += """
                            d0 = jp0 - ip0 + {OX};
                            d1 = jp1 - ip1 + {OY};
                            d2 = jp2 - ip2 + {OZ};
                            r2 = d0*d0 + d1*d1 + d2*d2;
                            r = sqrt(r2);
                            tmp_inner_phi += 0.5 * iq * jq / r;

                    """.format(
                        OXI=oxi,
                        OX=ox[0] * E,
                        OY=ox[1] * E,
                        OZ=ox[2] * E
                    )
        
        
        header = r"""
        #include <math.h>
        #define INT64 int64_t
        #define REAL double
        """

        src = r"""
        
        extern "C" int nearest_direct(
            const INT64 N,
            const REAL * RESTRICT P,
            const REAL * RESTRICT Q,
            REAL * RESTRICT phi
        ){{

            REAL tmp_phi = 0.0;

            #pragma omp parallel for reduction(+:tmp_phi)
            for(INT64 ix=0 ; ix<N ; ix++){{
                REAL tmp_inner_phi = 0.0;
                
                const REAL iq = Q[ix];
                const REAL ip0 = P[3*ix + 0];
                const REAL ip1 = P[3*ix + 1];
                const REAL ip2 = P[3*ix + 2];

                for(INT64 jx=(ix+1) ; jx<N ; jx++){{
                    
                    const REAL jq = Q[jx];
                    const REAL jp0 = P[3*jx + 0];
                    const REAL jp1 = P[3*jx + 1];
                    const REAL jp2 = P[3*jx + 2];

                    REAL d0 = ip0 - jp0;
                    REAL d1 = ip1 - jp1;
                    REAL d2 = ip2 - jp2;
                    
                    REAL r2 = d0*d0 + d1*d1 + d2*d2;
                    REAL r = sqrt(r2);

                    tmp_inner_phi += iq * jq / r;

                }}

                for(INT64 jx=0 ; jx<N ; jx++){{
                    
                    const REAL jq = Q[jx];
                    const REAL jp0 = P[3*jx + 0];
                    const REAL jp1 = P[3*jx + 1];
                    const REAL jp2 = P[3*jx + 2];

                    REAL d0;
                    REAL d1;
                    REAL d2;
                    
                    REAL r2;
                    REAL r;

                    {INNER}

                }}
                
                tmp_phi += tmp_inner_phi;

            }}
           
            phi[0] = tmp_phi;
            return 0;
        }}
        """.format(
            INNER=inner
        )

        self._lib = build.simple_lib_creator(header_code=header, src_code=src, name="kmc_fmm_nearest_direct")['nearest_direct']


    def __call__(self, N, P, Q):

        phi = ctypes.c_double(0)

        self._lib(
            INT64(N),
            P.ctypes.get_as_parameter(),
            Q.ctypes.get_as_parameter(),
            ctypes.byref(phi)
        )
        
        return phi.value


class PBCDirect:
    def __init__(self, E, domain, L):
        
        self.lrc = LongRangeMTL(L, domain)

        self._nd = NearestDirect(E)

        self.ncomp = 2*(L**2)
        self.half_ncomp = L**2

        self._lee = LocalExpEval(L)
        self.multipole_exp = np.zeros(self.ncomp, dtype=REAL)
        self.local_dot_coeffs = np.zeros(self.ncomp, dtype=REAL)

    def __call__(self, N, P, Q):

        sr = self._nd(N, P, Q)

        self.multipole_exp.fill(0)
        self.local_dot_coeffs.fill(0)
 
        for px in range(N):
            # multipole expansion for the whole cell
            self._lee.multipole_exp(
                spherical(tuple(P[px,:])),
                Q[px, 0],
                self.multipole_exp
            )
            # dot product for the local expansion for the cell
            self._lee.dot_vec(
                spherical(tuple(P[px,:])),
                Q[px, 0],
                self.local_dot_coeffs
            )

        L_tmp = np.zeros_like(self.local_dot_coeffs)
        self.lrc(self.multipole_exp, L_tmp)

        lr = 0.5 * np.dot(L_tmp, self.local_dot_coeffs)

        return sr + lr


