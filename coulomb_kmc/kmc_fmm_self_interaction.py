__author__ = "W.R.Saunders"


import ctypes
import numpy as np

REAL = ctypes.c_double
INT64 = ctypes.c_int64

from coulomb_kmc.common import BCType, PROFILE, cell_offsets

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


class FMMSelfInteraction:
    """
    Handles the self interaction between the new position and old position of a
    proposed move.

    :arg fmm: PyFMM instance to use.
    :arg domain: domain to use.
    :arg boundary_condition: Boundary condition of the KMC instance.
    :arg local_exp_eval: LocalExpEval instance to use for expansion manipulation.
    :arg mirror_direction: Mirror direction tuple for Dirichlet boundary conditions (default None).
    """

    def __init__(self, fmm, domain, boundary_condition, local_exp_eval, mirror_direction=None):

        assert boundary_condition in \
            (BCType.PBC, BCType.FREE_SPACE, BCType.NEAREST, BCType.FF_ONLY)

        self.domain = domain
        self._lee = local_exp_eval
        self._bc = boundary_condition
        self.fmm = fmm

        self._new27direct = 0.0
        ex = self.domain.extent
        for ox in cell_offsets:
            # image of old pos
            dox = np.array((ex[0] * ox[0], ex[1] * ox[1], ex[2] * ox[2]))
            if ox != (0,0,0):
                self._new27direct -= 1.0 / np.linalg.norm(dox)
        
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


        if self._bc == BCType.FREE_SPACE:
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
        
        if self._bc == BCType.FF_ONLY:
            ff_only_block = 'energy27 = 0.0;'
        else:
            ff_only_block = ''


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

                    {ff_only_block}
                    
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
            mirror_preloop=mirror_preloop,
            ff_only_block=ff_only_block
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
        pass

    def accept(self, movedata):
        pass

    def propose(self, total_movs, num_particles, host_data, cuda_data, arr, use_python=False):
        """
        Propose a move using the coulomb_kmc internal proposed move data structures.
        For details see `coulomb_kmc.kmc_mpi_decomp.FMMMPIDecomp.setup_propose_with_dats`.
        """

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




