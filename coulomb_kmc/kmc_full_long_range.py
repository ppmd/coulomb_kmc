__author__ = "W.R.Saunders"


import ctypes
import numpy as np
from math import *

REAL = ctypes.c_double
INT64 = ctypes.c_int64

# cuda imports if possible
import ppmd
import ppmd.cuda
from ppmd.coulomb.sph_harm import *
from ppmd.lib.build import simple_lib_creator
from coulomb_kmc.common import spherical, cell_offsets
from ppmd.coulomb.fmm_pbc import LongRangeMTL


class FullLongRangeEnergy:
    def __init__(self, L, domain, local_exp_eval, mirror_direction=None):
        # this should be a full PBC fmm instance
        self.domain = domain
        self._lee = local_exp_eval
        self.mirror_direction = mirror_direction
        self.L = L
        self.ncomp = 2*(L**2)
        self.half_ncomp = L**2
        
        self.lrc = LongRangeMTL(L, domain)


        self.multipole_exp = np.zeros(self.ncomp, dtype=REAL)
        self.local_dot_coeffs = np.zeros(self.ncomp, dtype=REAL)

        self._host_lib = self._init_host_lib()
        self._nthread = ppmd.runtime.NUM_THREADS
        assert self._nthread > 0
        # these get resized if needed
        self._thread_space = [np.zeros(4000, dtype=REAL) for tx in range(self._nthread)]
        self._thread_ptrs = np.array([tx.ctypes.get_as_parameter().value for tx in self._thread_space],
            dtype=ctypes.c_void_p)


    def _resize_if_needed(self, max_nprop):
        """
        //         NCOMP
        // nprop * NCOMP
        //         NCOMP
        // nprop * NCOMP
        // nprop * NCOMP
        """
        ncomp = (self.L**2)*2
        needed_space = 3 * max_nprop * ncomp + 2 * ncomp

        if self._thread_space[0].shape[0] < needed_space:
            self._thread_space = [np.zeros(needed_space, dtype=REAL) for tx in range(self._nthread)]
            self._thread_ptrs = np.array([tx.ctypes.get_as_parameter().value for tx in self._thread_space],
                dtype=ctypes.c_void_p)


    def initialise(self, positions, charges):
        assert self.domain.comm.size == 1, "need to MPI reduce coefficients"

        self.multipole_exp.fill(0)
        self.local_dot_coeffs.fill(0)

        for px in range(positions.npart_local):
            # multipole expansion for the whole cell
            self._lee.multipole_exp(
                spherical(tuple(positions[px,:])),
                charges[px, 0],
                self.multipole_exp
            )
            # dot product for the local expansion for the cell
            self._lee.dot_vec(
                spherical(tuple(positions[px,:])),
                charges[px, 0],
                self.local_dot_coeffs
            )
        
        L_tmp = np.zeros_like(self.local_dot_coeffs)
        self.lrc(self.multipole_exp, L_tmp)
        return 0.5 * np.dot(L_tmp, self.local_dot_coeffs)

    

    def propose(self, total_movs, num_particles, host_data, cuda_data, arr, use_python=False):
        
        if use_python:
            self.py_propose(total_movs, num_particles, host_data, cuda_data, arr)
        else:

            es = host_data['exclusive_sum']
            old_pos = host_data['old_positions']
            new_pos = host_data['new_positions']
            old_chr = host_data['old_charges']
            
            assert es.dtype == INT64
            assert old_pos.dtype == REAL
            assert old_chr.dtype == REAL
            assert new_pos.dtype == REAL
            assert arr.dtype == REAL

            shift_left = es[1:num_particles+1:]
            max_nprop = np.max(shift_left - es[0:num_particles:])
            self._resize_if_needed(max_nprop)

            L_tmp = np.zeros(self.ncomp, dtype=REAL)
            # get current long range energy
            self.lrc(self.multipole_exp, L_tmp)
            old_energy = 0.5 * np.dot(L_tmp, self.local_dot_coeffs)

            self._host_lib(
                REAL(old_energy),
                INT64(arr.shape[1]),
                INT64(num_particles),
                es.ctypes.get_as_parameter(),
                old_pos.ctypes.get_as_parameter(),
                old_chr.ctypes.get_as_parameter(),
                new_pos.ctypes.get_as_parameter(),
                self.multipole_exp.ctypes.get_as_parameter(),
                self.local_dot_coeffs.ctypes.get_as_parameter(),
                self.lrc.linop_data.ctypes.get_as_parameter(),
                self.lrc.linop_indptr.ctypes.get_as_parameter(),
                self.lrc.linop_indices.ctypes.get_as_parameter(),
                self._thread_ptrs.ctypes.get_as_parameter(),
                arr.ctypes.get_as_parameter()
            )


    def py_propose(self, total_movs, num_particles, host_data, cuda_data, arr, use_python=True):
        es = host_data['exclusive_sum']
        old_pos = host_data['old_positions']
        new_pos = host_data['new_positions']
        old_chr = host_data['old_charges']
        
        assert es.dtype == INT64
        assert old_pos.dtype == REAL
        assert old_chr.dtype == REAL
        assert new_pos.dtype == REAL
        assert arr.dtype == REAL
        
        # tmp vars

        to_remove = np.zeros(self.ncomp, dtype=REAL)
        prop_mexp = np.zeros_like(to_remove)

        to_remove_dot_vec = np.zeros_like(to_remove)
        dot_vec = np.zeros_like(to_remove)

        L_tmp = np.zeros_like(to_remove)
        # get current long range energy
        self.lrc(self.multipole_exp, L_tmp)
        old_energy = 0.5 * np.dot(L_tmp, self.local_dot_coeffs)

        for px in  range(num_particles):
            # assumed charge doesn't change
            charge = old_chr[px]
            opos = old_pos[px, :]

            nprop = es[px+1, 0] - es[px, 0]

            # remove old multipole expansion coeffs
            to_remove.fill(0)
            self._lee.multipole_exp(spherical(tuple(opos)), -charge, to_remove)
            
            # remove dot product coeffs
            to_remove_dot_vec.fill(0)
            self._lee.dot_vec(spherical(tuple(opos)), -charge, to_remove_dot_vec)


            for movxi, movx in enumerate(range(es[px, 0], es[px+1, 0])):
                prop_mexp[:] = self.multipole_exp[:].copy()
                npos = new_pos[movx, :]

                # compute the mutipole expansion of the proposed config
                self._lee.multipole_exp(spherical(tuple(npos)), charge, prop_mexp)
                # remove the old pos
                prop_mexp[:] += to_remove

                # do the same for the dot product vector
                dot_vec[:] = self.local_dot_coeffs.copy()
                dot_vec[:] += to_remove_dot_vec[:]

                # add on the proposed position
                self._lee.dot_vec(spherical(tuple(npos)), charge, dot_vec)
                
                # apply long range mtl
                L_tmp.fill(0)
                self.lrc(prop_mexp, L_tmp)
                
                # compute long range energy contribution
                new_energy = 0.5 * np.dot(L_tmp, dot_vec)

                arr[px, movxi] += old_energy - new_energy


    def accept(self, movedata):

        realdata = movedata[:7].view(dtype=REAL)

        old_position = realdata[0:3:]
        new_position = realdata[3:6:]
        charge       = realdata[6]

        # modify the multipole expansion for the coarest level
        self._lee.multipole_exp(spherical(tuple(old_position)), -charge, self.multipole_exp)
        self._lee.multipole_exp(spherical(tuple(new_position)),  charge, self.multipole_exp)

        # modify the dot product coefficients
        self._lee.dot_vec(spherical(tuple(old_position)), -charge, self.local_dot_coeffs)
        self._lee.dot_vec(spherical(tuple(new_position)),  charge, self.local_dot_coeffs)

    def eval_field(self, points, out):
        npoints = points.shape[0]
        lexp = np.zeros(self.ncomp, REAL)
        self.lrc(self.multipole_exp, lexp)

        for px in range(npoints):
            pointx = points[px, :]
            lr_tmp = self._lee.compute_phi_local(lexp, spherical(tuple(pointx)))[0]
            out[px] += lr_tmp

    def _init_host_lib(self):
        ncomp = (self.L**2)*2
        half_ncomp = self.L**2
        def _re_lm(l, m): return l**2 + l + m


        mirror_preloop = ''
        mirror_loop_0 = ''
        if self.mirror_direction is not None:
            # convert mirror directions to coefficients
            mcoeff = dict()
            mcoeff['mcoeffx'] = -1.0 if self.mirror_direction[0] else 1.0 
            mcoeff['mcoeffy'] = -1.0 if self.mirror_direction[1] else 1.0 
            mcoeff['mcoeffz'] = -1.0 if self.mirror_direction[2] else 1.0 

            mirror_preloop += '''
            const REAL mopx = opx * {mcoeffx};
            const REAL mopy = opy * {mcoeffy};
            const REAL mopz = opz * {mcoeffz};

            REAL moradius, motheta, mophi;
            spherical(mopx, mopy, mopz, &moradius, &motheta, &mophi);
            multipole_exp(charge, moradius, motheta, mophi, old_moments);
            
            // remove the contribs for the old mirror position
            local_dot_vec(charge, moradius, motheta, mophi, old_evector);
            '''.format(**mcoeff)

            mirror_loop_0 = '''
            const REAL mnpx = npx * {mcoeffx};
            const REAL mnpy = npy * {mcoeffy};
            const REAL mnpz = npz * {mcoeffz};

            REAL mnradius, mntheta, mnphi;
            spherical(mnpx, mnpy, mnpz, &mnradius, &mntheta, &mnphi);

            // add on the new moments
            multipole_exp(-1.0*charge, mnradius, mntheta, mnphi, &new_moments[movii*NCOMP]);

            // add on the new evector coefficients
            local_dot_vec(-1.0*charge, mnradius, mntheta, mnphi, &new_evector[movii*NCOMP]);
            '''.format(**mcoeff)


        src = r'''
        
        {MULTIPOLE_HEADER}
        {MULTIPOLE_SRC}

        {LOCAL_EVAL_HEADER}
        {LOCAL_EVAL_SRC}

        {EVEC_HEADER}
        {EVEC_SRC}



        static inline void apply_dipole_correction(
            const REAL * RESTRICT M,
            REAL * RESTRICT L
        ){{
            L[RE_1P1] += DIPOLE_SX * M[RE_1P1];
            L[RE_1N1] += DIPOLE_SX * M[RE_1P1];

            L[IM_1P1] -= DIPOLE_SY * M[IM_1P1];
            L[IM_1N1] += DIPOLE_SY * M[IM_1P1];

            L[RE_1_0] += DIPOLE_SZ * M[RE_1_0];

            return;
        }}
        
        static inline REAL dot_product(
            const REAL * RESTRICT A,
            const REAL * RESTRICT B
        ){{
            REAL tmp = 0.0;
            for(int cx=0 ; cx<NCOMP ; cx++){{
                tmp += A[cx] * B[cx];
            }}
            return tmp;
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

        static inline void linop_csr(
            const REAL * RESTRICT linop_data,
            const INT64 * RESTRICT linop_indptr,
            const INT64 * RESTRICT linop_indices,
            const REAL * RESTRICT x,
            REAL * RESTRICT b
        ){{
            
            INT64 data_ind = 0;
            for(INT64 row=0 ; row<HALF_NCOMP ; row++){{
                REAL row_tmp = 0.0;
                for(INT64 col_ind=linop_indptr[row] ; col_ind<linop_indptr[row+1] ; col_ind++){{
                    const INT64 col = linop_indices[data_ind];
                    const REAL data = linop_data[data_ind];
                    data_ind++;
                    row_tmp += data * x[col];
                }}
                b[row] = row_tmp;
            }}
            return;
        }}


        extern "C" int long_range_energy(
            const REAL old_energy,
            const INT64 store_stride,
            const INT64 num_particles,
            const INT64 * RESTRICT exclusive_sum,
            const REAL * RESTRICT old_positions,
            const REAL * RESTRICT old_charges,
            const REAL * RESTRICT new_positions,
            const REAL * RESTRICT existing_multipole,
            const REAL * RESTRICT existing_evector,
            const REAL * RESTRICT linop_data,
            const INT64 * RESTRICT linop_indptr,
            const INT64 * RESTRICT linop_indices,
            REAL * RESTRICT * RESTRICT tmp_space,
            REAL * RESTRICT out
        )
        {{

            #pragma omp parallel for schedule(dynamic)
            for(INT64 px=0 ; px<num_particles ; px++){{

                const REAL charge = old_charges[px];
                const REAL opx = old_positions[3*px + 0];
                const REAL opy = old_positions[3*px + 1];
                const REAL opz = old_positions[3*px + 2];

                const INT64 nprop = exclusive_sum[px+1] - exclusive_sum[px];
 
                const int threadid =  omp_get_thread_num();

                REAL * RESTRICT old_evector = tmp_space[threadid];              // NCOMP
                REAL * RESTRICT new_evector = old_evector + NCOMP;              // nprop * NCOMP
                REAL * RESTRICT old_moments = new_evector + nprop*NCOMP;        // NCOMP
                REAL * RESTRICT new_moments = old_moments + NCOMP;              // nprop * NCOMP
                REAL * RESTRICT new_local_moments = new_moments + nprop*NCOMP;  // nprop * NCOMP

                // copy the existing moments
                for(int nx=0 ; nx<NCOMP ; nx++){{ old_moments[nx] = existing_multipole[nx]; }}

                // add on the multipole expansion for the old position
                REAL oradius, otheta, ophi;
                spherical(opx, opy, opz, &oradius, &otheta, &ophi);
                multipole_exp(-1.0 * charge, oradius, otheta, ophi, old_moments);

                // copy the existing evector
                for(int nx=0 ; nx<NCOMP ; nx++){{ old_evector[nx] = existing_evector[nx]; }}
                
                // remove the contribs for the old position
                local_dot_vec(-1.0 * charge, oradius, otheta, ophi, old_evector);
                
                // Do the above for mirror charge if required.
                {MIRROR_PRELOOP}

                // loop over the proposed new positions and copy old positions and compute spherical coordinate
                // vectors
                #pragma omp simd simdlen(8)
                for(INT64 movii=0 ; movii<nprop ; movii++){{
                    const INT64 movi = movii + exclusive_sum[px];
                    const REAL npx = new_positions[3*movi + 0];
                    const REAL npy = new_positions[3*movi + 1];
                    const REAL npz = new_positions[3*movi + 2];

                    REAL nradius, ntheta, nphi;
                    spherical(npx, npy, npz, &nradius, &ntheta, &nphi);

                    // copy the old position moments
                    for(int nx=0 ; nx<NCOMP ; nx++){{ new_moments[movii*NCOMP + nx] = old_moments[nx]; }}

                    // add on the new moments
                    multipole_exp(charge, nradius, ntheta, nphi, &new_moments[movii*NCOMP]);

                    // copy the old evector coefficients
                    for(int nx=0 ; nx<NCOMP ; nx++){{ new_evector[movii*NCOMP + nx] = old_evector[nx]; }}

                    // add on the new evector coefficients
                    local_dot_vec(charge, nradius, ntheta, nphi, &new_evector[movii*NCOMP]);

                    {MIRROR_LOOP_0}
                }}

                #pragma omp simd simdlen(8)
                for(INT64 movii=0 ; movii<nprop ; movii++){{

                    // apply the lin op to the real part
                    linop_csr(linop_data, linop_indptr, linop_indices,
                        &new_moments[movii*NCOMP], &new_local_moments[movii*NCOMP]);

                    // then the imaginary part
                    linop_csr(linop_data, linop_indptr, linop_indices,
                        &new_moments[movii*NCOMP + HALF_NCOMP], &new_local_moments[movii*NCOMP + HALF_NCOMP]);

                    // apply the dipole correction
                    apply_dipole_correction(&new_moments[movii*NCOMP], &new_local_moments[movii*NCOMP]);

                }}
                
                #pragma omp simd simdlen(8)
                for(INT64 movii=0 ; movii<nprop ; movii++){{
                    
                    // apply dot product to each proposed move to get energy
                    
                    const REAL new_energy = 0.5 * dot_product(
                        &new_local_moments[movii*NCOMP], &new_evector[movii*NCOMP]);

                    out[px*store_stride + movii] += old_energy - new_energy;

                }}


            }}

            return 0;
        }}
        '''.format(
            MULTIPOLE_HEADER=self._lee.create_multipole_header,
            MULTIPOLE_SRC=self._lee.create_multipole_src,
            LOCAL_EVAL_HEADER=self._lee.create_local_eval_header,
            LOCAL_EVAL_SRC=self._lee.create_local_eval_src,
            EVEC_HEADER=self._lee.create_dot_vec_header,
            EVEC_SRC=self._lee.create_dot_vec_src,
            MIRROR_PRELOOP=mirror_preloop,
            MIRROR_LOOP_0=mirror_loop_0
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
            ))
        )
        
        return simple_lib_creator(header_code=header, src_code=src)['long_range_energy']
