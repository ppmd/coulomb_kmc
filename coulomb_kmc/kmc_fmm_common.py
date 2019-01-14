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

from coulomb_kmc.common import BCType, PROFILE

# cuda imports if possible
import ppmd
import ppmd.cuda
from ppmd.coulomb.sph_harm import *
from ppmd.lib.build import simple_lib_creator
#from ppmd.coulomb.fmm_pbc import DipoleCorrector


if ppmd.cuda.CUDA_IMPORT:
    cudadrv = ppmd.cuda.cuda_runtime.cudadrv
    # the device should be initialised already by ppmd
    from pycuda.compiler import SourceModule
    import pycuda.gpuarray as gpuarray

def spherical(xyz):
    """
    Converts the cartesian coordinates in xyz to spherical coordinates
    (radius, polar angle, longitude angle)
    """
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

cell_offsets = (
    ( -1, -1, -1),
    (  0, -1, -1),
    (  1, -1, -1),
    ( -1,  0, -1),
    (  0,  0, -1),
    (  1,  0, -1),
    ( -1,  1, -1),
    (  0,  1, -1),
    (  1,  1, -1),

    ( -1, -1,  0),
    (  0, -1,  0),
    (  1, -1,  0),
    ( -1,  0,  0),
    (  0,  0,  0),
    (  1,  0,  0),
    ( -1,  1,  0),
    (  0,  1,  0),
    (  1,  1,  0),

    ( -1, -1,  1),
    (  0, -1,  1),
    (  1, -1,  1),
    ( -1,  0,  1),
    (  0,  0,  1),
    (  1,  0,  1),
    ( -1,  1,  1),
    (  0,  1,  1),
    (  1,  1,  1),
)


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


class FullLongRangeEnergy:
    def __init__(self, fmm, local_exp_eval):
        # this should be a full PBC fmm instance
        self.fmm = fmm
        self.domain = fmm.domain
        self._lee = local_exp_eval
        L = self.fmm.L
        self.ncomp = 2*(L**2)
        self.half_ncomp = L**2
        self.lrc = LongRangeCorrection(fmm, fmm.domain, local_exp_eval)

        self.multipole_exp = np.zeros(self.ncomp, dtype=REAL)
        self.local_dot_coeffs = np.zeros(self.ncomp, dtype=REAL)

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
        self.lrc.apply_lr_mtl(self.multipole_exp, L_tmp)
        return 0.5 * np.dot(L_tmp, self.local_dot_coeffs)

    def propose(self, total_movs, num_particles, host_data, cuda_data, arr, use_python=True):
        es = host_data['exclusive_sum']
        old_pos = host_data['old_positions']
        new_pos = host_data['new_positions']
        old_chr = host_data['old_charges']
        
        assert es.dtype == INT64
        assert old_pos.dtype == REAL
        assert old_chr.dtype == REAL
        assert new_pos.dtype == REAL
        assert arr.dtype == REAL

        assert use_python
        
        # tmp vars

        to_remove = np.zeros(self.ncomp, dtype=REAL)
        prop_mexp = np.zeros_like(to_remove)

        to_remove_dot_vec = np.zeros_like(to_remove)
        dot_vec = np.zeros_like(to_remove)

        L_tmp = np.zeros_like(to_remove)


        # get current long range energy
        self.lrc.apply_lr_mtl(self.multipole_exp, L_tmp)
        old_energy = 0.5 * np.dot(L_tmp, self.local_dot_coeffs)

        print("old_energy", old_energy)
        

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
                self.lrc.apply_lr_mtl(prop_mexp, L_tmp)
                
                # compute long range energy contribution
                new_energy = 0.5 * np.dot(L_tmp, dot_vec)

                arr[px, movxi] += old_energy - new_energy
                print("new lr contrib", new_energy - old_energy)




class LongRangeCorrection:
    def __init__(self, fmm, domain, local_exp_eval):

        self.fmm = fmm
        self.domain = domain
        self._lee = local_exp_eval
        self._rvec = self.fmm._boundary_terms

        L = self.fmm.L
        self.ncomp = 2*(L**2)
        self.half_ncomp = L**2
        self.rmat = np.zeros((self.half_ncomp, self.half_ncomp), dtype=REAL)
        #self.rmat = csr_matrix((self.half_ncomp, self.half_ncomp), dtype=REAL)

        row = 0
        for jx in range(L):
            for kx in range(-jx, jx+1):
                col = 0
                for nx in range(L):
                    for mx in range(-nx, nx+1):
                        if (not abs(mx-kx) > jx+nx) and \
                            (not (abs(mx-kx) % 2 == 1)) and \
                            (not (abs(jx+nx) % 2 == 1)):
                            
                            self.rmat[row, col] = _h(jx, kx, nx, mx) * self._rvec[_re_lm(jx+nx, mx-kx)]
                        col += 1
                row += 1

        self.sparse_rmat = csr_matrix(self.rmat)
        self.linop = aslinearoperator(self.rmat)
        self.sparse_linop = aslinearoperator(self.sparse_rmat)
        self._linop_data = np.array(self.sparse_rmat.data, dtype=REAL)
        self._linop_indptr = np.array(self.sparse_rmat.indptr, dtype=INT64)
        self._linop_indices = np.array(self.sparse_rmat.indices, dtype=INT64)

        self._orig_space = np.zeros((self.ncomp), dtype=REAL)

        self._prop_space = np.zeros((100, self.ncomp), dtype=REAL)
        self._prop_matmul = np.zeros((self.ncomp, 100), dtype=REAL)

        self._host_lib = self._init_host_lib()
        self._nthread = ppmd.runtime.NUM_THREADS
        assert self._nthread > 0

        self._thread_space = [np.zeros(4000, dtype=REAL) for tx in range(self._nthread)]
        self._thread_ptrs = np.array([tx.ctypes.get_as_parameter().value for tx in self._thread_space],
            dtype=ctypes.c_void_p)

        # dipole correction vars
        # self._dpc = DipoleCorrector(L, self.domain.extent, self.fmm._lr_mtl_func)

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

    def apply_lr_mtl(self, M, L):
        L[:self.half_ncomp] = self.sparse_linop.dot(M[:self.half_ncomp])
        L[self.half_ncomp:] = self.sparse_linop.dot(M[self.half_ncomp:])
        self.fmm.dipole_corrector(M, L)


    def compute_corrections(self, total_movs, num_particles, host_data, cuda_data, lr_correction, arr):

        es = host_data['exclusive_sum']
        old_pos = host_data['old_positions']
        new_pos = host_data['new_positions']
        old_chr = host_data['old_charges']
        
        L = self.fmm.L
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
            self._linop_data.ctypes.get_as_parameter(),
            self._linop_indptr.ctypes.get_as_parameter(),
            self._linop_indices.ctypes.get_as_parameter(),
            self._thread_ptrs.ctypes.get_as_parameter(),
            arr.ctypes.get_as_parameter()
        )

    def compute_corrections_py(self, total_movs, num_particles, host_data, cuda_data, lr_correction, arr):

        es = host_data['exclusive_sum']
        old_pos = host_data['old_positions']
        new_pos = host_data['new_positions']
        old_chr = host_data['old_charges']
        
        L = self.fmm.L
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
        ncomp = (self.fmm.L**2)*2
        half_ncomp = self.fmm.L**2

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
                #pragma omp simd simdlen(8)
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
    def __init__(self, fmm, domain, boundary_condition, local_exp_eval):
        self.domain = domain
        self._lee = local_exp_eval
        self._bc = boundary_condition
        self.fmm = fmm

        self._ncomp = (self.fmm.L**2)*2
        self._half_ncomp = self.fmm.L**2
        self._lr_correction = np.zeros(self._ncomp, dtype=REAL)
        self._lr_correction_local = np.zeros_like(self._lr_correction)

        if self._bc is BCType.PBC:
            self.lrc = LongRangeCorrection(fmm, domain, local_exp_eval)

        self._new27direct = 0.0
        ex = self.domain.extent
        for ox in cell_offsets:
            # image of old pos
            dox = np.array((ex[0] * ox[0], ex[1] * ox[1], ex[2] * ox[2]))
            if ox != (0,0,0):
                self._new27direct -= 1.0 / np.linalg.norm(dox)

        self._existing_correction = _SILocalEvaluator(self._lee)
        

        preloop = ''
        bc27 = ''
        if self._bc not in (BCType.NEAREST, BCType.PBC):
            pass
        else:
            bc27 = 'energy27 = (DOMAIN_27_ENERGY);\n'
            for oxi, ox in enumerate(cell_offsets):

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

                bc27 += '''
                const REAL dpx{oxi} = dox{oxi} + opx;
                const REAL dpy{oxi} = doy{oxi} + opy;
                const REAL dpz{oxi} = doz{oxi} + opz;

                const REAL ddx{oxi} = dpx{oxi} - npx;
                const REAL ddy{oxi} = dpy{oxi} - npy;
                const REAL ddz{oxi} = dpz{oxi} - npz;

                energy27 += 1.0 / sqrt(ddx{oxi}*ddx{oxi} + ddy{oxi}*ddy{oxi} + ddz{oxi}*ddz{oxi});
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

                    REAL tmp_energy = energy27;
                    out[store_stride * px + movii] = coeff * tmp_energy;
                    
                }}

            }}

            return 0;
        }}
        '''.format(
            bc27=bc27,
            preloop=preloop
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
            ))
        )
        

        self.lib = simple_lib_creator(header_code=header, src_code=src)['self_interaction']

    def initialise(self):
        self._lr_correction[:] = 0.0
        self._lr_correction_local[:] = 0.0


    def accept(self, movedata):

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

        assign_gen = 'double rhol = 1.0;\n'
        for lx in range(self.L):
            for mx in range(-lx, lx+1):
                assign_gen += 'out[{ind}] += {ylmm} * rhol * charge;\n'.format(
                        ind=cube_ind(lx, mx),
                        ylmm=str(sph_gen.get_y_sym(lx, mx)[0])
                    )
                assign_gen += 'out[IM_OFFSET + {ind}] += (-1.0) * {ylmm} * rhol * charge;\n'.format(
                        ind=cube_ind(lx, mx),
                        ylmm=str(sph_gen.get_y_sym(lx, mx)[1])
                    )
            assign_gen += 'rhol *= radius;\n'

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
            assign_gen += 'rhol *= iradius;\n'

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



class LocalOctalBase:

    def _profile_inc(self, key, inc):
        key = self.__class__.__name__ + ':' + key
        if key not in PROFILE.keys():
            PROFILE[key] = inc
        else:
            PROFILE[key] += inc

    def _get_fmm_cell(self, ix, cell_map, slow_to_fast=False):
        # produces xyz tuple by default
        cc = cell_map[ix][0]
        cx, cy, cz = self._cell_lin_to_tuple(cc)

        if not slow_to_fast:
            return cx, cy, cz
        else:
            return cz, cy, cx
    
    def _cell_lin_to_tuple(self, cc):

        R = self.fmm.R
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
        return cx, cy, cz


    def _global_cell_xyz(self, tcx):
        """get global cell index from xyz tuple"""
        gcs = self.global_cell_size
        return tcx[0] + gcs[0] * ( tcx[1] + gcs[1] * tcx[2] )



