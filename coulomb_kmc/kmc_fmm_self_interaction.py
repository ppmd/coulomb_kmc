__author__ = "W.R.Saunders"


import ctypes
import numpy as np
from math import *
import scipy
from scipy.special import lpmv
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import aslinearoperator

from functools import lru_cache

REAL = ctypes.c_double
INT64 = ctypes.c_int64

from coulomb_kmc.common import BCType, PROFILE, spherical, cell_offsets

# cuda imports if possible
import ppmd
import ppmd.cuda
from ppmd.coulomb.sph_harm import *
from ppmd.lib.build import simple_lib_creator
from ppmd.coulomb.fmm_pbc import DipoleCorrector, FMMPbc, LongRangeMTL

from coulomb_kmc.kmc_expansion_tools import LocalExpEval

if ppmd.cuda.CUDA_IMPORT:
    cudadrv = ppmd.cuda.cuda_runtime.cudadrv
    # the device should be initialised already by ppmd
    from pycuda.compiler import SourceModule
    import pycuda.gpuarray as gpuarray


@lru_cache(maxsize=128)
def _A(n,m):
    return ((-1.0)**n)/sqrt(factorial(n-m)*factorial(n+m))
@lru_cache(maxsize=128)
def _h(j,k,n,m):
    if abs(k) > j: return 0.0
    if abs(m) > n: return 0.0
    if abs(m-k) > j+n: return 0.0
    icoeff = ((1.j)**(abs(k-m) - abs(k) - abs(m))).real
    return icoeff * _A(n, m) * _A(j, k) / (((-1.0) ** n) * _A(j+n, m-k))

def _re_lm(l, m): return l**2 + l + m


class LongRangeCorrection:
    def __init__(self, L, domain, local_exp_eval):
        
        self.L = L
        self._lee = local_exp_eval
        self.lr_mtl = LongRangeMTL(L, domain)
        self.domain = domain

        self.ncomp = 2*(L**2)
        self.half_ncomp = L**2

        self._orig_space = np.zeros((self.ncomp), dtype=REAL)

        self._prop_space = np.zeros((100, self.ncomp), dtype=REAL)
        self._prop_matmul = np.zeros((self.ncomp, 100), dtype=REAL)

        self._host_lib = self._init_host_lib()


        self._nthread = ppmd.runtime.NUM_THREADS
        assert self._nthread > 0
        # these get resized if needed
        self._thread_space = [np.zeros(4000, dtype=REAL) for tx in range(self._nthread)]
        self._thread_ptrs = np.array([tx.ctypes.get_as_parameter().value for tx in self._thread_space],
            dtype=ctypes.c_void_p)

    def _resize_if_needed_py(self, nc):
        if self._prop_space.shape[0] < nc:
            self._prop_space = np.zeros((nc, self.ncomp), dtype=REAL)
            self._prop_matmul = np.zeros((self.ncomp, nc), dtype=REAL)
    
    def _resize_if_needed(self, max_nprop):
        ncomp = (self.fmm.L**2)*2
        needed_space = max_nprop * (ncomp*3 + 3)
        if self._thread_space[0].shape[0] < needed_space:
            self._thread_space = [np.zeros(needed_space, dtype=REAL) for tx in range(self._nthread)]
            self._thread_ptrs = np.array([tx.ctypes.get_as_parameter().value for tx in self._thread_space],
                dtype=ctypes.c_void_p)


    def compute_corrections(self, total_movs, num_particles, host_data, cuda_data, lr_correction, arr):

        es = host_data['exclusive_sum']
        old_pos = host_data['old_positions']
        new_pos = host_data['new_positions']
        old_chr = host_data['old_charges']
        
        L = self.L
        ncomp = (L**2)*2
        half_ncomp = (L**2)

        shift_left = es[1:num_particles+1:]
        max_nprop = np.max(shift_left - es[0:num_particles:])
        self._resize_if_needed(max_nprop)

        self._host_lib(
            INT64(arr.shape[1]),
            INT64(num_particles),
            es.ctypes.get_as_parameter(),
            old_pos.ctypes.get_as_parameter(),
            old_chr.ctypes.get_as_parameter(),
            new_pos.ctypes.get_as_parameter(),
            lr_correction.ctypes.get_as_parameter(),
            self.lr_mtl.linop_data.ctypes.get_as_parameter(),
            self.lr_mtl.linop_indptr.ctypes.get_as_parameter(),
            self.lr_mtl.linop_indices.ctypes.get_as_parameter(),
            self._thread_ptrs.ctypes.get_as_parameter(),
            arr.ctypes.get_as_parameter()
        )

    def compute_corrections_py(self, total_movs, num_particles, host_data, cuda_data, lr_correction, arr):

        es = host_data['exclusive_sum']
        old_pos = host_data['old_positions']
        new_pos = host_data['new_positions']
        old_chr = host_data['old_charges']
        
        L = self.L
        ncomp = (L**2)*2
        half_ncomp = (L**2)
        
        assert es.dtype == INT64
        assert old_pos.dtype == REAL
        assert old_chr.dtype == REAL
        assert new_pos.dtype == REAL
        assert arr.dtype == REAL

        for px in range(num_particles):
            charge = old_chr[px, 0]
            opos = old_pos[px, :]
            num_prop = es[px+1, 0] - es[px, 0]
            
            disp = spherical(tuple(opos))

            self._orig_space[:] = lr_correction[:].copy()
            # self._orig_space[:] = 0.0
            self._lee.multipole_exp(disp, -1.0 * charge, self._orig_space)
            self._resize_if_needed_py(num_prop)

            prop_pos = new_pos[es[px, 0]:es[px+1, 0]:, :]
            prop_sph = spherical(prop_pos)

            self._prop_space.fill(0.0)

            for movxi in range(num_prop):
                self._lee.multipole_exp(prop_sph[movxi, :], charge, self._prop_space[movxi, :])
                self._prop_space[movxi, :] += self._orig_space[:].copy()
            
            self._prop_matmul[:half_ncomp ,:num_prop] = \
                self.sparse_linop.matmat(np.transpose(self._prop_space[:num_prop,:half_ncomp]))
            self._prop_matmul[half_ncomp: ,:num_prop] = \
                self.sparse_linop.matmat(np.transpose(self._prop_space[:num_prop,half_ncomp:]))

            self._prop_space[:num_prop, :] = self._prop_matmul[:, :num_prop].transpose().copy()

            for movxi in range(num_prop):
                arr[px, movxi] -= charge * self._lee.compute_phi_local(
                    self._prop_space[movxi, :], prop_sph[movxi, :]
                )[0]


    def _init_host_lib(self):
        ncomp = (self.L**2)*2
        half_ncomp = self.L**2

        src = r'''
        
        {MULTIPOLE_HEADER}
        {MULTIPOLE_SRC}

        {LOCAL_EVAL_HEADER}
        {LOCAL_EVAL_SRC}

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


        extern "C" int long_range_self_interaction(
            const INT64 store_stride,
            const INT64 num_particles,
            const INT64 * RESTRICT exclusive_sum,
            const REAL * RESTRICT old_positions,
            const REAL * RESTRICT old_charges,
            const REAL * RESTRICT new_positions,
            const REAL * RESTRICT lr_correction,
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

                REAL * RESTRICT old_moments = tmp_space[threadid];
                REAL * RESTRICT sph_vectors = old_moments + NCOMP;
                REAL * RESTRICT new_moments = sph_vectors + nprop*3;
                REAL * RESTRICT new_local_moments = new_moments + nprop*NCOMP;

                // copy the existing correction
                for(int nx=0 ; nx<NCOMP ; nx++){{ old_moments[nx] = lr_correction[nx]; }}
 
                //printf("clr lr correct %f %f %f %f\n", old_moments[0], old_moments[1], old_moments[2], old_moments[3]);

                // add on the multipole expansion for the old position
                REAL oradius, otheta, ophi;
                spherical(opx, opy, opz, &oradius, &otheta, &ophi);
                multipole_exp(-1.0 * charge, oradius, otheta, ophi, old_moments);

                // loop over the proposed new positions and copy old positions and compute spherical coordinate
                // vectors
                for(INT64 movii=0 ; movii<nprop ; movii++){{
                    const INT64 movi = movii + exclusive_sum[px];
                    const REAL npx = new_positions[3*movi + 0];
                    const REAL npy = new_positions[3*movi + 1];
                    const REAL npz = new_positions[3*movi + 2];

                    REAL nradius, ntheta, nphi;
                    spherical(npx, npy, npz, &nradius, &ntheta, &nphi);
                    sph_vectors[movii*3 + 0] = nradius;
                    sph_vectors[movii*3 + 1] = ntheta;
                    sph_vectors[movii*3 + 2] = nphi;

                    // copy the old position moments
                    for(int nx=0 ; nx<NCOMP ; nx++){{ new_moments[movii*NCOMP + nx] = old_moments[nx]; }}

                    // add on the new moments
                    multipole_exp(charge, nradius, ntheta, nphi, &new_moments[movii*NCOMP]);
                }}

                //printf("clr multi exp %f %f %f %f\n", new_moments[0], new_moments[1], new_moments[2], new_moments[3]);
                #pragma omp simd simdlen(8)
                for(INT64 movii=0 ; movii<nprop ; movii++){{

                    // apply the lin op to the real part
                    linop_csr(linop_data, linop_indptr, linop_indices,
                        &new_moments[movii*NCOMP], &new_local_moments[movii*NCOMP]);

                    // then the imaginary part
                    linop_csr(linop_data, linop_indptr, linop_indices,
                        &new_moments[movii*NCOMP + HALF_NCOMP], &new_local_moments[movii*NCOMP + HALF_NCOMP]);

                }}

                //printf("clr local exp %f %f %f %f\n", new_local_moments[0], new_local_moments[1], new_local_moments[2], new_local_moments[3]);

                #pragma omp simd simdlen(8)
                for(INT64 movii=0 ; movii<nprop ; movii++){{
                    REAL outdouble = 0.0;
                    const REAL nradius = sph_vectors[movii*3 + 0];
                    const REAL ntheta  = sph_vectors[movii*3 + 1];
                    const REAL nphi    = sph_vectors[movii*3 + 2];

                    local_eval(nradius, ntheta, nphi, &new_local_moments[movii*NCOMP], &outdouble);
                    out[px*store_stride + movii] -= charge * outdouble;
                }}


            }}

            return 0;
        }}
        '''.format(
            MULTIPOLE_HEADER=self._lee.create_multipole_header,
            MULTIPOLE_SRC=self._lee.create_multipole_src,
            LOCAL_EVAL_HEADER=self._lee.create_local_eval_header,
            LOCAL_EVAL_SRC=self._lee.create_local_eval_src
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
            ))
        )
        
        return simple_lib_creator(header_code=header, src_code=src)['long_range_self_interaction']




class FMMSelfInteraction:
    def __init__(self, fmm, domain, boundary_condition, local_exp_eval, mirror_direction=None):
        self.domain = domain
        self._lee = local_exp_eval
        self._bc = boundary_condition
        self.fmm = fmm

        self._ncomp = (self.fmm.L**2)*2
        self._half_ncomp = self.fmm.L**2
        self._lr_correction = np.zeros(self._ncomp, dtype=REAL)
        self._lr_correction_local = np.zeros_like(self._lr_correction)

        if self._bc is BCType.PBC:
            self.lrc = LongRangeCorrection(self.fmm.L, domain, local_exp_eval)

        self._new27direct = 0.0
        ex = self.domain.extent
        for ox in cell_offsets:
            # image of old pos
            dox = np.array((ex[0] * ox[0], ex[1] * ox[1], ex[2] * ox[2]))
            if ox != (0,0,0):
                self._new27direct -= 1.0 / np.linalg.norm(dox)

        self._existing_correction = _SILocalEvaluator(self._lee)
        
        mirror_block = ''
        mirror_preloop = ''

        if self._bc == BCType.FREE_SPACE:
            co = ((0,0,0),)
        else:
            co = cell_offsets


        if mirror_direction is not None:
            # convert mirror directions to coefficients
            mcoeff = dict()

            mcoeff['mcoeffx'] = -1.0 if mirror_direction[0] else 1.0 
            mcoeff['mcoeffy'] = -1.0 if mirror_direction[1] else 1.0 
            mcoeff['mcoeffz'] = -1.0 if mirror_direction[2] else 1.0 

            # compute position of old mirror charge
            mirror_preloop += '''
            const REAL mopx = opx * {mcoeffx};
            const REAL mopy = opy * {mcoeffy};
            const REAL mopz = opz * {mcoeffz};
            '''.format(**mcoeff)

            mirror_block += '''
            const REAL mnpx = npx * {mcoeffx};
            const REAL mnpy = npy * {mcoeffy};
            const REAL mnpz = npz * {mcoeffz};
            '''.format(**mcoeff)


            for oxi, ox in enumerate(co):

                oxi_zero = 0 if (ox[0] == 0 and ox[1] == 0 and ox[2] == 0) else 1
                oxv = {
                    'oxi': str(oxi),
                    'oxi_zero': str(oxi_zero)
                }
                oxv.update(mcoeff)

                mirror_block += '''

                // offset of the old charge
                const REAL mdpx{oxi} = dox{oxi} + mopx;
                const REAL mdpy{oxi} = doy{oxi} + mopy;
                const REAL mdpz{oxi} = doz{oxi} + mopz;
                
                // diff to the old mirror in offset
                const REAL mddx{oxi} = mdpx{oxi} - npx;
                const REAL mddy{oxi} = mdpy{oxi} - npy;
                const REAL mddz{oxi} = mdpz{oxi} - npz;
                
                // remove old energy
                energy27 -= 2.0 / sqrt(mddx{oxi}*mddx{oxi} + mddy{oxi}*mddy{oxi} + mddz{oxi}*mddz{oxi});
                
                
                // offset of the new charge
                const REAL mnpx{oxi} = dox{oxi} + mnpx;
                const REAL mnpy{oxi} = doy{oxi} + mnpy;
                const REAL mnpz{oxi} = doz{oxi} + mnpz;

                // diff to the new mirror in the offset
                const REAL mnddx{oxi} = mnpx{oxi} - npx;
                const REAL mnddy{oxi} = mnpy{oxi} - npy;
                const REAL mnddz{oxi} = mnpz{oxi} - npz;

                // add on the new contrib
                energy27 += 1.0 / sqrt(mnddx{oxi}*mnddx{oxi} + mnddy{oxi}*mnddy{oxi} + mnddz{oxi}*mnddz{oxi});

                // the factor 2 required for b_bp with the non-mirrors
                energy27 += o_bbp{oxi};


                // compute b_b, first with non-mirror
                const REAL do_opx{oxi} = opx - dpx{oxi};
                const REAL do_opy{oxi} = opy - dpy{oxi};
                const REAL do_opz{oxi} = opz - dpz{oxi};
                energy27 -= ({oxi_zero} == 0) ? 0.0 : 1.0 / sqrt(do_opx{oxi}*do_opx{oxi} + do_opy{oxi}*do_opy{oxi} + do_opz{oxi}*do_opz{oxi});
                
                // with the mirror
                const REAL do_mopx{oxi} = opx - mdpx{oxi};
                const REAL do_mopy{oxi} = opy - mdpy{oxi};
                const REAL do_mopz{oxi} = opz - mdpz{oxi};
                energy27 += 1.0 / sqrt(do_mopx{oxi}*do_mopx{oxi} + do_mopy{oxi}*do_mopy{oxi} + do_mopz{oxi}*do_mopz{oxi});
                

                '''.format(**oxv)



        preloop = ''
        bc27 = ''

        for oxi, ox in enumerate(co):

            preloop += '''
            const REAL dox{oxi} = EX * {OX};
            const REAL doy{oxi} = EY * {OY};
            const REAL doz{oxi} = EZ * {OZ};
            '''.format(
                oxi=str(oxi),
                OX=str(ox[0]),
                OY=str(ox[1]),
                OZ=str(ox[2]),
            )


        if self._bc not in (BCType.NEAREST, BCType.PBC):
            pass
        else:
            bc27 = 'energy27 = (DOMAIN_27_ENERGY);\n'
            for oxi, ox in enumerate(cell_offsets):

                bc27 += '''
                const REAL dpx{oxi} = dox{oxi} + opx;
                const REAL dpy{oxi} = doy{oxi} + opy;
                const REAL dpz{oxi} = doz{oxi} + opz;

                const REAL ddx{oxi} = dpx{oxi} - npx;
                const REAL ddy{oxi} = dpy{oxi} - npy;
                const REAL ddz{oxi} = dpz{oxi} - npz;
                
                const REAL o_bbp{oxi} = 1.0 / sqrt(ddx{oxi}*ddx{oxi} + ddy{oxi}*ddy{oxi} + ddz{oxi}*ddz{oxi});
                energy27 += o_bbp{oxi};
                '''.format(
                    oxi=str(oxi)
                )

        src = r'''
        extern "C" int self_interaction(
            const INT64 store_stride,
            const INT64 num_particles,
            const INT64 * RESTRICT exclusive_sum,
            const REAL * RESTRICT old_positions,
            const REAL * RESTRICT old_charges,
            const REAL * RESTRICT new_positions,
            REAL * RESTRICT out
        )
        {{
            
            {preloop}

            #pragma omp parallel for schedule(dynamic)
            for(INT64 px=0 ; px<num_particles ; px++){{
                
                const REAL coeff = old_charges[px] * old_charges[px];
                const REAL opx = old_positions[3*px + 0];
                const REAL opy = old_positions[3*px + 1];
                const REAL opz = old_positions[3*px + 2];

                {mirror_preloop}

                const INT64 nprop = exclusive_sum[px+1] - exclusive_sum[px];

                #pragma omp simd simdlen(8)
                for(INT64 movii=0 ; movii<nprop ; movii++){{
                    const INT64 movi = movii + exclusive_sum[px];
                    const REAL npx = new_positions[3*movi + 0];
                    const REAL npy = new_positions[3*movi + 1];
                    const REAL npz = new_positions[3*movi + 2];
                    
                    const REAL dx = opx - npx;
                    const REAL dy = opy - npy;
                    const REAL dz = opz - npz;
                    
                    REAL energy27 = (1.0 / sqrt(dx*dx + dy*dy + dz*dz));

                    {bc27}

                    {mirror_block}
                    
                    REAL tmp_energy = energy27;
                    out[store_stride * px + movii] = coeff * tmp_energy;
                    
                }}

            }}

            return 0;
        }}
        '''.format(
            bc27=bc27,
            preloop=preloop,
            mirror_block=mirror_block,
            mirror_preloop=mirror_preloop
        )

        header = str(
            Module((
                Include('stdio.h'),
                Include('math.h'),
                Define('DOMAIN_27_ENERGY', str(self._new27direct)),
                Define('INT64', 'int64_t'),
                Define('REAL', 'double'),
                Define('EX', str(self.domain.extent[0])),
                Define('EY', str(self.domain.extent[1])),
                Define('EZ', str(self.domain.extent[2])),
                Define('PRINTF(A,B,C)', r'printf("%s:\t%f,\t%s:\t%f,\t%s:\t%f\n", #A, A, #B, B, #C, C);'),
                Define('PRINTF1(A)', r'printf("%s:\t%f\n", #A, A);'),
            ))
        )
        
        #print(src)
        self.lib = simple_lib_creator(header_code=header, src_code=src)['self_interaction']

    def initialise(self):
        self._lr_correction[:] = 0.0
        self._lr_correction_local[:] = 0.0


    def accept(self, movedata):
        
        # not needed with "full" long-range
        return

        realdata = movedata[:7].view(dtype=REAL)

        old_position = realdata[0:3:]
        new_position = realdata[3:6:]
        charge       = realdata[6]
        if self._bc is BCType.PBC:
            # compute the long-range corrections to the mutlipole expansion for further proposals
            self._lee.multipole_exp(spherical(tuple(old_position)), -1.0*charge, self._lr_correction)
            self._lee.multipole_exp(spherical(tuple(new_position)), charge, self._lr_correction)

            self._lr_correction_local[:self._half_ncomp] = \
                self.lrc.sparse_linop(self._lr_correction[:self._half_ncomp]).copy()
            self._lr_correction_local[self._half_ncomp:] = \
                self.lrc.sparse_linop(self._lr_correction[self._half_ncomp:]).copy()


    def propose(self, total_movs, num_particles, host_data, cuda_data, arr, use_python=False):

        es = host_data['exclusive_sum']
        old_pos = host_data['old_positions']
        new_pos = host_data['new_positions']
        old_chr = host_data['old_charges']
        
        assert es.dtype == INT64
        assert old_pos.dtype == REAL
        assert old_chr.dtype == REAL
        assert new_pos.dtype == REAL
        assert arr.dtype == REAL

        self.lib(
            INT64(arr.shape[1]),
            INT64(num_particles),
            es.ctypes.get_as_parameter(),
            old_pos.ctypes.get_as_parameter(),
            old_chr.ctypes.get_as_parameter(),
            new_pos.ctypes.get_as_parameter(),
            arr.ctypes.get_as_parameter()           
        )

        # this code is redundant if using the "full" approach to long-range energy
        return
        if self._bc is BCType.PBC:
            print("warning always using python")
            use_python = True


            if not use_python:
                self._existing_correction(num_particles, es, old_pos, old_chr, self._lr_correction_local, arr)
                self.lrc.compute_corrections(total_movs, num_particles, host_data, cuda_data,
                    self._lr_correction, arr)
            else:
                for px in  range(num_particles):
                    # assumed charge doesn't change
                    charge = old_chr[px]
                    opos = old_pos[px, :]
                    nprop = es[px+1, 0] - es[px, 0]
                    if nprop > 0:
                        existing_correction = self._lee.compute_phi_local(self._lr_correction_local,
                            spherical(tuple(opos)))[0] * charge
                        arr[px, :nprop] += existing_correction

                
                for px in range(num_particles):
                    # assumed charge doesn't change
                    charge = old_chr[px]
                    opos = old_pos[px, :]

                    for movxi, movx in enumerate(range(es[px, 0], es[px+1, 0])):
                        tmp = self.py_self_interaction_indirect(opos, new_pos[movx, :], charge)
                        arr[px, movxi] -= tmp
                        print("old lr diff contrib", tmp)



    def py_self_interaction_direct(self, old_pos, prop_pos, q):
        """
        Compute the self interaction of the proposed move in the primary image with the old position
        in all other images.
        """

        ex = self.domain.extent

        # self interaction with primary image
        if self._bc is BCType.FREE_SPACE:
            return q * q / np.linalg.norm(old_pos - prop_pos)
        
        # 26 nearest primary images
        elif self._bc in (BCType.NEAREST, BCType.PBC):
            coeff = q * q
            e_tmp = self._new27direct * coeff

            for ox in cell_offsets:
                # image of old pos
                dox = np.array((ex[0] * ox[0], ex[1] * ox[1], ex[2] * ox[2]))
                iold_pos = old_pos + dox
                e_tmp +=  coeff / np.linalg.norm(iold_pos - prop_pos)

            return e_tmp

        else:
            raise RuntimeError('bad boundary condition in _self_interaction')

    def py_self_interaction_indirect(self, old_pos, prop_pos, q):
        ex = self.domain.extent
        lexp = self._really_long_range_diff(old_pos, prop_pos, q)
        # the origin is the centre of the domain hence no offsets are needed
        disp = spherical(tuple(prop_pos))
        rlr = q * self._lee.compute_phi_local(lexp, disp)[0]
        return rlr



    def _multipole_diff(self, old_pos, new_pos, charge, arr):
        # output is in the numpy array arr
        # plan is to do all "really long range" corrections
        # as a matmul

        # remove the old charge
        disp = spherical(tuple(old_pos))
        self._lee.multipole_exp(disp, -1.0 * charge, arr)
        
        # add the new charge
        disp = spherical(tuple(new_pos))
        self._lee.multipole_exp(disp, charge, arr)


    def _really_long_range_diff(self, old_pos, prop_pos, q):
        """
        Compute the correction in potential field from the "very well separated"
        images
        """
        
        l2 = self.fmm.L * self.fmm.L * 2
        arr = np.zeros(l2)
        arr_out = np.zeros(l2)

        arr[:] = self._lr_correction[:].copy()
        #print("plr lr correct", arr[:4])

        self._multipole_diff(old_pos, prop_pos, q, arr)
        
        #print("plr multi exp", arr[:4])

        # use the really long range part of the fmm instance (extract this into a matrix)
        self.fmm._translate_mtl_lib['mtl_test_wrapper'](
            INT64(self.fmm.L),
            REAL(1.),
            arr.ctypes.get_as_parameter(),
            self.fmm._boundary_ident.ctypes.get_as_parameter(),
            self.fmm._boundary_terms.ctypes.get_as_parameter(),
            self.fmm._a.ctypes.get_as_parameter(),
            self.fmm._ar.ctypes.get_as_parameter(),
            self.fmm._ipower_mtl.ctypes.get_as_parameter(),
            arr_out.ctypes.get_as_parameter()
        )
        
        # this probably breaks equal-and-opposite somehow
        #self.fmm.dipole_corrector(arr, arr_out)

        # print("plr local exp", arr_out[:4])
        return arr_out



class _SILocalEvaluator:
    def __init__(self, local_exp_eval):
        lee = local_exp_eval

        src = r'''
        #define REAL double
        #define INT64 int64_t

        {LEE_SRC}

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

        extern "C" int si_local_eval(
            const INT64 num_particles,
            const INT64 out_stride,
            const INT64 * RESTRICT exclusive_sum,
            const REAL  * RESTRICT old_positions,
            const REAL  * RESTRICT old_charges,
            const REAL  * RESTRICT local_expansion,
            REAL * RESTRICT out
        ){{
            #pragma omp parallel for schedule(dynamic)
            for(INT64 px=0 ; px<num_particles ; px++){{

                REAL radius, theta, phi;
                spherical(
                    old_positions[px*3 + 0],
                    old_positions[px*3 + 1],
                    old_positions[px*3 + 2],
                    &radius, &theta, &phi
                );

                // assume local_eval does a += on energy
                REAL energy = 0.0;

                local_eval(radius, theta, phi, local_expansion, &energy);
                energy *= old_charges[px];

                const INT64 es_start = exclusive_sum[px];
                const INT64 es_end = exclusive_sum[px+1];
                const INT64 es_count = es_end - es_start;
                for(INT64 ex=0 ; ex<es_count ; ex++){{
                    out[px*out_stride + ex] += energy;
                }}
            }}
            return 0;
        }}

        '''.format(
            LEE_SRC=lee.create_local_eval_src
        )

        header = lee.create_local_eval_header

        self._lib = simple_lib_creator(header, src)['si_local_eval']


    def __call__(self, num_particles, exclusive_sum, old_positions, old_charges,
            local_expansion, out):
        
        assert out.dtype == ctypes.c_double
        assert exclusive_sum.dtype == INT64
        assert old_positions.dtype == ctypes.c_double
        assert old_charges.dtype == ctypes.c_double
        assert local_expansion.dtype == ctypes.c_double

        return self._lib(
            INT64(num_particles),
            INT64(out.shape[1]),
            exclusive_sum.ctypes.get_as_parameter(),
            old_positions.ctypes.get_as_parameter(),
            old_charges.ctypes.get_as_parameter(),
            local_expansion.ctypes.get_as_parameter(),
            out.ctypes.get_as_parameter()
        )



