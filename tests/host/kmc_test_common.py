

import ctypes
import ppmd
from ppmd.lib import build

REAL = ctypes.c_double
INT64 = ctypes.c_int64


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

                    tmp_inner_phi += Q[ix] * Q[jx] / r;

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









