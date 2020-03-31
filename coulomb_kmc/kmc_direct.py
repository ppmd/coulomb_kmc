

import ctypes
import ppmd
from ppmd.lib import build

REAL = ctypes.c_double
INT64 = ctypes.c_int64

from itertools import product

from coulomb_kmc.common import spherical, BCType
from ppmd.coulomb.fmm_pbc import LongRangeMTL
from coulomb_kmc.kmc_expansion_tools import LocalExpEval


from ppmd.coulomb.direct import FreeSpaceDirect, NearestDirect, PBCDirect, FarFieldDirect

import numpy as np

from cgen import *

from ppmd import access



class PairDirectFromDats:

    def __init__(self, domain, boundary_condition, L, max_num_groups, mirror_mode=False, energy_unit=1.0):

        self.domain = domain
        self.bc = boundary_condition
        self.L = L

        exp_eval = LocalExpEval(L)

        self.lrc = LongRangeMTL(L, domain)

        ncomp = (self.L**2)*2
        half_ncomp = self.L**2
        def _re_lm(l, m): return l**2 + l + m
        E = self.domain.extent[0]
        
        assert abs(E - self.domain.extent[0]) < 10.**-14
        assert abs(E - self.domain.extent[1]) < 10.**-14
        assert abs(E - self.domain.extent[2]) < 10.**-14

        m_quater_extent_z = -0.25 * self.domain.extent[2]
        

        if mirror_mode:
            group_decl = r"""
            const REAL bb_positions[12] = {{
                positions[ix*3 + 0],
                positions[ix*3 + 1],
                positions[ix*3 + 2] + {Z_SHIFT},
                group_positions[ix * MAX_NUM_GROUPS * 3 + gx * 3 + 0],
                group_positions[ix * MAX_NUM_GROUPS * 3 + gx * 3 + 1],
                group_positions[ix * MAX_NUM_GROUPS * 3 + gx * 3 + 2] + {Z_SHIFT},
                positions[ix*3 + 0],
                positions[ix*3 + 1],
                -1.0 * (positions[ix*3 + 2] + {Z_SHIFT}),
                group_positions[ix * MAX_NUM_GROUPS * 3 + gx * 3 + 0],
                group_positions[ix * MAX_NUM_GROUPS * 3 + gx * 3 + 1],
                -1.0 * (group_positions[ix * MAX_NUM_GROUPS * 3 + gx * 3 + 2] + {Z_SHIFT})
            }};
            const REAL bb_charges[4] = {{
                charges[ix],
                group_charges[ix*MAX_NUM_GROUPS+gx],
                -1.0 * charges[ix],
                -1.0 * group_charges[ix*MAX_NUM_GROUPS+gx]                
            }};
            const INT64 NG = 4;
            """.format(
                Z_SHIFT=m_quater_extent_z
            )
        else:
            group_decl = r"""
            const REAL bb_positions[6] = {
                positions[ix*3 + 0],
                positions[ix*3 + 1],
                positions[ix*3 + 2],
                group_positions[ix * MAX_NUM_GROUPS * 3 + gx * 3 + 0],
                group_positions[ix * MAX_NUM_GROUPS * 3 + gx * 3 + 1],
                group_positions[ix * MAX_NUM_GROUPS * 3 + gx * 3 + 2]
            };
            const REAL bb_charges[2] = {
                charges[ix],
                group_charges[ix*MAX_NUM_GROUPS+gx]
            };
            const INT64 NG = 2;
            //printf("MAX_NUM_GROUPS %d, zpos %f gx %d\n", MAX_NUM_GROUPS, group_positions[ix * MAX_NUM_GROUPS * 3 + gx * 3 + 2], gx);
            //for(int ix=0 ; ix<NG ; ix++){
            //    printf("%d | P %f %f %f Q %f\n", ix, bb_positions[ix*3], bb_positions[ix*3+1], bb_positions[ix*3+2], bb_charges[ix]);
            //}



            """


        inner_direct = ''
        if boundary_condition in (BCType.NEAREST, BCType.PBC):
            ox_range = tuple(range(-1, 2))
            for oxi, ox in enumerate(product(ox_range, ox_range, ox_range)):
                    if ox[0] != 0 or ox[1] != 0 or ox[2] != 0:
                        inner_direct += """
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


        pbc_call = ''
        if boundary_condition == BCType.PBC:
            pbc_call = r"""
            const REAL tmp_energy_lr = pbc_direct(N, positions, charges, linop_data, linop_indptr, linop_indices);
            tmp_energy += tmp_energy_lr;
            """



        src = r"""

        static inline REAL nearest_direct(
            const INT64 N,
            const REAL * RESTRICT P,
            const REAL * RESTRICT Q
        ){{

            REAL tmp_phi = 0.0;

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

                    {INNER_DIRECT}

                }}
                
                tmp_phi += tmp_inner_phi;

            }}
           
            return tmp_phi;
        }}


        static inline void spherical(
            const REAL dx, const REAL dy, const REAL dz,
            REAL *radius, REAL *theta, REAL *phi
        ){{
            const REAL dx2 = dx*dx;
            const REAL dx2_p_dy2 = dx2 + dy*dy;
            const REAL d2 = dx2_p_dy2 + dz*dz;
            *radius = sqrt(d2);
            *theta = atan2(sqrt(dx2_p_dy2), dz);
            *phi = atan2(dy, dx);       
            return;
        }}
        
        {MULTIPOLE_HEADER}
        {MULTIPOLE_SRC}

        {EVEC_HEADER}
        {EVEC_SRC}


        static inline REAL linop_csr_both(
            const REAL * RESTRICT linop_data,
            const INT64 * RESTRICT linop_indptr,
            const INT64 * RESTRICT linop_indices,
            const REAL * RESTRICT x1,
            const REAL * RESTRICT E
        ){{
            
            INT64 data_ind = 0;
            REAL dot_tmp = 0.0;

            for(INT64 row=0 ; row<HALF_NCOMP ; row++){{

                REAL row_tmp_1 = 0.0;
                REAL row_tmp_2 = 0.0;

                for(INT64 col_ind=linop_indptr[row] ; col_ind<linop_indptr[row+1] ; col_ind++){{
                    const INT64 col = linop_indices[data_ind];
                    const REAL data = linop_data[data_ind];
                    data_ind++;
                    row_tmp_1 += data * x1[col];
                    row_tmp_2 += data * x1[col  + HALF_NCOMP];
                }}

                dot_tmp += row_tmp_1 * E[row] + row_tmp_2 * E[row + HALF_NCOMP];
            }}

            return dot_tmp;
        }}


        static inline REAL apply_dipole_correction_split(
            const REAL * RESTRICT M,
            const REAL * RESTRICT E
        ){{
            
            REAL tmp = 0.0;

            tmp += (DIPOLE_SX * M[RE_1P1]) * E[RE_1P1];
            tmp += (DIPOLE_SX * M[RE_1P1]) * E[RE_1N1];
        
            tmp -= (DIPOLE_SY * M[IM_1P1]) * E[IM_1P1];
            tmp += (DIPOLE_SY * M[IM_1P1]) * E[IM_1N1];

            tmp += (DIPOLE_SZ * M[RE_1_0]) * E[RE_1_0];

            return tmp;
        }}


        static inline REAL pbc_direct(
            const INT64            N,
            const REAL  * RESTRICT positions,
            const REAL  * RESTRICT charges,
            const REAL  * RESTRICT linop_data,
            const INT64 * RESTRICT linop_indptr,
            const INT64 * RESTRICT linop_indices
        ){{

            REAL new_moments[NCOMP];
            REAL new_evector[NCOMP];
            
            for(int cx=0 ; cx<NCOMP ; cx++){{
                new_moments[cx] = 0.0;
                new_evector[cx] = 0.0;
            }}

            for(int ix=0 ; ix<N ; ix++){{
                REAL radius, theta, phi;
                const REAL px = positions[ix*3 + 0];
                const REAL py = positions[ix*3 + 1];
                const REAL pz = positions[ix*3 + 2];
                const REAL ch = charges[ix];

                spherical(px, py, pz, &radius, &theta, &phi);

                local_dot_vec(ch, radius, theta, phi, new_evector);
                multipole_exp(ch, radius, theta, phi, new_moments);

            }}


            REAL new_energy = 0.5 * linop_csr_both(
                linop_data, linop_indptr, linop_indices,
                new_moments,
                new_evector
            );

            new_energy += 0.5 * apply_dipole_correction_split(
                new_moments,
                new_evector
            );


            return new_energy;
        }}





        static inline REAL compute_energy(
            const INT64            N,
            const REAL  * RESTRICT positions,
            const REAL  * RESTRICT charges,
            const REAL  * RESTRICT linop_data,
            const INT64 * RESTRICT linop_indptr,
            const INT64 * RESTRICT linop_indices
        ){{
            REAL tmp_energy = nearest_direct(N, positions, charges);
                   
            {PBC_CALL}

            return tmp_energy;
        }}
        
        extern "C" int direct_from_dats(
            const INT64            N,
            const INT64 * RESTRICT flags,
            const REAL  * RESTRICT positions,
            const REAL  * RESTRICT charges,
            const INT64 * RESTRICT group_counts,
            const REAL  * RESTRICT group_positions,
            const REAL  * RESTRICT group_charges,
            const REAL  * RESTRICT linop_data,
            const INT64 * RESTRICT linop_indptr,
            const INT64 * RESTRICT linop_indices,
                  REAL  * RESTRICT group_energy
        ){{
            
            #pragma omp parallel for
            for(INT64 ix=0 ; ix<N ; ix++ ){{
                if ( flags[ix] > 0 ){{
                    for(INT64 gx=0 ; gx<group_counts[ix] ; gx++ ){{

                        {GROUP_DECL}
                        
                        group_energy[ix*MAX_NUM_GROUPS+gx] = compute_energy(
                            NG,
                            bb_positions,
                            bb_charges,
                            linop_data,
                            linop_indptr,
                            linop_indices
                        ) * {ENERGY_UNIT};

                    }}
                }}
            }}
            return 0;
        }}
        """.format(
            GROUP_DECL=group_decl,
            INNER_DIRECT=inner_direct,
            PBC_CALL=pbc_call,
            MULTIPOLE_HEADER=exp_eval.create_multipole_header,
            MULTIPOLE_SRC=exp_eval.create_multipole_src,
            EVEC_HEADER=exp_eval.create_dot_vec_header,
            EVEC_SRC=exp_eval.create_dot_vec_src,
            ENERGY_UNIT=float(energy_unit)
        )
        

        header = str(
            Module((
                Include('omp.h'),
                Include('stdio.h'),
                Include('math.h'),
                Define('INT64', 'int64_t'),
                Define('REAL', 'double'),
                Define('NCOMP', str(ncomp)),
                Define('HALF_NCOMP', str(half_ncomp)),
                Define('DIPOLE_SX', str(self.lrc.dipole_correction[0])),
                Define('DIPOLE_SY', str(self.lrc.dipole_correction[1])),
                Define('DIPOLE_SZ', str(self.lrc.dipole_correction[2])),
                Define('RE_1P1', str(_re_lm(1, 1))),
                Define('RE_1_0', str(_re_lm(1, 0))),
                Define('RE_1N1', str(_re_lm(1,-1))),
                Define('IM_1P1', str(_re_lm(1, 1) + half_ncomp)),
                Define('IM_1_0', str(_re_lm(1, 0) + half_ncomp)),
                Define('IM_1N1', str(_re_lm(1,-1) + half_ncomp)),
                Define('MAX_NUM_GROUPS', str(max_num_groups))
            ))
        )

        self._lib = build.simple_lib_creator(header, src)['direct_from_dats']
    

    def __call__(self,
        flags, 
        positions, 
        charges, 
        group_counts, 
        group_positions,
        group_charges,
        group_energy
    ):  
        N = INT64(positions.npart_local)
        
        assert flags.dtype == INT64 
        assert positions.dtype == REAL
        assert charges.dtype == REAL
        assert group_counts.dtype == INT64
        assert group_positions.dtype == REAL
        assert group_charges.dtype == REAL
        assert group_energy.dtype == REAL

        self._lib(
            N,
            flags.ctypes_data_access(access.READ, pair=False),
            positions.ctypes_data_access(access.READ, pair=False),
            charges.ctypes_data_access(access.READ, pair=False),
            group_counts.ctypes_data_access(access.READ, pair=False),
            group_positions.ctypes_data_access(access.READ, pair=False),
            group_charges.ctypes_data_access(access.READ, pair=False),
            self.lrc.linop_data.ctypes.get_as_parameter(),
            self.lrc.linop_indptr.ctypes.get_as_parameter(),
            self.lrc.linop_indices.ctypes.get_as_parameter(),
            group_energy.ctypes_data_access(access.WRITE, pair=False)
        )

        group_energy.ctypes_data_post(access.WRITE)


















