from __future__ import division, print_function, absolute_import
__author__ = "W.R.Saunders"

"""
Octal Tree classes for kmc
"""

import numpy as np
from math import ceil
import ctypes
from itertools import product
from cgen import *

from ppmd import mpi
from ppmd.coulomb.sph_harm import SphGen, SphSymbol, cmplx_mul
from ppmd.lib import build

from coulomb_kmc.common import BCType, PROFILE
from coulomb_kmc.kmc_fmm_common import LocalOctalBase, LocalExpEval, spherical

MPI = mpi.MPI

REAL = ctypes.c_double
INT64 = ctypes.c_int64

class LocalCellExpansions(LocalOctalBase):
    """
    Object to get, store and update local expansions from an fmm instance.
    """
    def __init__(self, mpi_decomp):
        
        self.md = mpi_decomp
        self.fmm = self.md.fmm
        self.domain = self.md.domain
        self.comm = self.md.comm
        self.fmm = self.md.fmm
        self.local_store_dims = self.md.local_store_dims
        self.local_size = self.md.local_size
        self.local_offset = self.md.local_offset        
        self.cell_indices = self.md.cell_indices
        self.cell_offsets = self.md.cell_offsets
        self.global_cell_size = self.md.global_cell_size
        self.entry_local_size = self.md.entry_local_size
        self.entry_local_offset = self.md.entry_local_offset
        self.periodic_factors = self.md.periodic_factors
        self.global_to_local = self.md.global_to_local
        self.boundary_condition = self.md.boundary_condition
        self._lee = LocalExpEval(self.fmm.L)

        local_store_dims = self.local_store_dims

        self.local_expansions = np.zeros(
            local_store_dims + [2 * (self.fmm.L**2)], dtype=REAL)
        self.remote_inds = np.zeros(local_store_dims + [1], dtype=INT64)
        self.remote_inds[:] = -1
        
        self._wing = MPI.Win()
        data_nbytes = self.fmm.tree_plain[-1][0,0,0,:].nbytes
        self._win = self._wing.Create(
            self.fmm.tree_plain[-1], disp_unit=data_nbytes, comm=self.comm)

        gmap_nbytes = self.fmm.tree[-1].global_to_local[0,0,0].nbytes
        self._win_ind = self._wing.Create(
            self.fmm.tree[-1].global_to_local,
            disp_unit=gmap_nbytes,
            comm=self.comm
        )
        
        self._ncomp = (self.fmm.L**2)*2

        # compute the cell centres
        self.cell_centres = np.zeros(local_store_dims + [3], dtype=REAL)
        self._host_lib = self._init_host_kernels(self.fmm.L)


    def accept(self, movedata):
        self._accept_py(movedata)

    def _accept_py(self, movedata):
        realdata = movedata[:7].view(dtype=REAL)

        old_position = realdata[0:3:]
        new_position = realdata[3:6:]
        charge       = realdata[6]
        gid          = movedata[7]
        old_fmm_cell = movedata[8]
        new_fmm_cell = movedata[9]

        new_cell_tuple = self._cell_lin_to_tuple(new_fmm_cell)
        old_cell_tuple = self._cell_lin_to_tuple(old_fmm_cell)
        
        old_tuple_s2f = tuple(reversed(old_cell_tuple))
        new_tuple_s2f = tuple(reversed(new_cell_tuple))

        lsd = self.local_store_dims

        if self.boundary_condition is BCType.FREE_SPACE:

            def old_cell_well_separated(cx):
                return not all([abs(cx - old_tuple_s2f[cxi]) < 2 for cxi, cx in enumerate(cx)])
            
            lsdi = (range(lsd[0]), range(lsd[1]), range(lsd[2]))

            for lsx in product(*lsdi):
                cellx = (self.cell_indices[0][lsx[0]], self.cell_indices[1][lsx[1]], 
                    self.cell_indices[2][lsx[2]])

                if old_cell_well_separated(cellx) and \
                    abs(self.periodic_factors[0][lsx[0]]) == 0 and \
                    abs(self.periodic_factors[1][lsx[1]]) == 0 and \
                    abs(self.periodic_factors[2][lsx[2]]) == 0:

                    cind = tuple(list(lsx) + [None])
                    cell_centre = self.cell_centres[cind][0]
                    disp = spherical(tuple(old_position - cell_centre))
                    self._lee.local_exp(disp, -1.0 * charge, self.local_expansions[cind])

            def new_cell_well_separated(cx):
                return not all([abs(cx - new_tuple_s2f[cxi]) < 2 for cxi, cx in enumerate(cx)])

            for lsx in product(*lsdi):
                cellx = (self.cell_indices[0][lsx[0]], self.cell_indices[1][lsx[1]], 
                    self.cell_indices[2][lsx[2]])

                if new_cell_well_separated(cellx) and \
                    abs(self.periodic_factors[0][lsx[0]]) == 0 and \
                    abs(self.periodic_factors[1][lsx[1]]) == 0 and \
                    abs(self.periodic_factors[2][lsx[2]]) == 0:

                    cind = tuple(list(lsx) + [None])
                    cell_centre = self.cell_centres[cind][0]
                    disp = spherical(tuple(new_position - cell_centre))
                    self._lee.local_exp(disp, charge, self.local_expansions[cind])

        else:
            raise RuntimeError()

    def propose(self, total_movs, num_particles, host_data, cuda_data):

        u0 = None
        u1 = None

        self._host_lib(
            INT64(num_particles),
            host_data['old_fmm_cells'].ctypes.get_as_parameter(),
            self.cell_centres.ctypes.get_as_parameter(),
            host_data['old_positions'].ctypes.get_as_parameter(),
            host_data['old_charges'].ctypes.get_as_parameter(),
            self.local_expansions.ctypes.get_as_parameter(),
            host_data['old_energy_i'].ctypes.get_as_parameter()
        )
        self._host_lib(
            INT64(total_movs),
            host_data['new_fmm_cells'].ctypes.get_as_parameter(),
            self.cell_centres.ctypes.get_as_parameter(),
            host_data['new_positions'].ctypes.get_as_parameter(),
            host_data['new_charges'].ctypes.get_as_parameter(),
            self.local_expansions.ctypes.get_as_parameter(),
            host_data['new_energy_i'].ctypes.get_as_parameter()
        )        

        u0 = host_data['old_energy_i'][:num_particles:]
        u1 = host_data['new_energy_i'][:total_movs:]

        return u0, u1


    def initialise(self, positions, charges, fmm_cells):
        self.positions = positions
        self.charges = charges
        self.fmm_cells = fmm_cells
        self._get_local_expansions()
        self._compute_cell_centres()
        self.group = self.positions.group


    def _compute_cell_centres(self):
        lsd = self.local_store_dims
        extent_s2f = list(reversed(self.domain.extent))
        ncell_s2f = self.global_cell_size
        widths = [ex/sx for ex, sx in zip(extent_s2f, ncell_s2f)]
        starts = [ -0.5*ex + 0.5*wx for wx, ex in zip(widths, extent_s2f)]
        centres = [[starts[dimi] + cx * widths[dimi] for cx in dimx] for \
            dimi, dimx in enumerate(self.cell_offsets)]

        # store the centres as xyz 
        for lcellx in product(range(lsd[0]), range(lsd[1]), range(lsd[2])):
            self.cell_centres[lcellx[0], lcellx[1], lcellx[2], :] = \
                (
                    centres[2][lcellx[2]],
                    centres[1][lcellx[1]], 
                    centres[0][lcellx[0]]
                )

    
    def _get_local_expansions(self):
        # copy/get the local expansions required from the fmm instance

        rank = self.comm.rank
        ncomp = (self.fmm.L ** 2) * 2

        remote_cells = []
        
        self.comm.Barrier()
        self._win_ind.Fence(MPI.MODE_NOPUT)
        
        lsd = self.local_store_dims
        lcl = self.cell_indices

        # these are slowest to fastest tuples of cells
        for local_cellx in product(range(lsd[0]), range(lsd[1]), range(lsd[2])):
            
            cellx = (lcl[0][local_cellx[0]], lcl[1][local_cellx[1]], lcl[2][local_cellx[2]])
            owning_rank = self.fmm.tree[-1].owners[cellx[0], cellx[1], cellx[2]]
            assert owning_rank > -1
            assert owning_rank < self.comm.size

            if rank == owning_rank:
                # can do a direct copy as this rank owns the local expansions in the fmm instance
                cellind = self.fmm.tree[-1].global_to_local[cellx[0], cellx[1], cellx[2]]
                self.local_expansions[local_cellx[0], local_cellx[1], local_cellx[2], :] = \
                        self.fmm.tree_plain[-1].ravel()[ cellind * ncomp: (cellind + 1) * ncomp :]

            else:
                # get the remote local index for this global index
                # these get calls are non blocking and are sync'd on the win fence call

                gcellx = self._global_cell_xyz((cellx[2], cellx[1], cellx[0]))

                self._win_ind.Get(self.remote_inds[local_cellx[0], local_cellx[1], local_cellx[2], :],
                        owning_rank, target=gcellx)

                remote_cells.append((cellx, gcellx, owning_rank, local_cellx))

        self._win_ind.Fence(MPI.MODE_NOPUT)
        
        self.comm.Barrier()           
        self._win.Fence(MPI.MODE_NOPUT)
        
        # get the remote data
        for cell_tup in remote_cells:
            
            cellx = cell_tup[0]
            gcellx = cell_tup[1]
            owning_rank = cell_tup[2]
            local_cellx = cell_tup[3]
            
            remote_ind = self.remote_inds[local_cellx[0], local_cellx[1], local_cellx[2], 0]
            assert remote_ind > -1

            self._win.Get(
                    self.local_expansions[local_cellx[0], local_cellx[1], local_cellx[2], :],
                    owning_rank, target=remote_ind)

        self._win.Fence(MPI.MODE_NOPUT)
        self.comm.Barrier()

    
    @staticmethod
    def _init_host_kernels(L):
        ncomp = (L**2)*2

        sph_gen = SphGen(maxl=L-1, theta_sym='theta', phi_sym='phi', ctype='double', avoid_calls=True)
        
        flops = dict(sph_gen.flops)

        def cube_ind(l, m):
            return ((l) * ( (l) + 1 ) + (m) )
        
        EC = ''
        for lx in range(L):
            EC += 'coeff = charge * rhol;\n'

            for mx in range(-lx, lx+1):
                smx = 'n' if mx < 0 else 'p'
                smx += str(abs(mx))

                re_lnm = SphSymbol('reln{lx}m{mx}'.format(lx=lx, mx=smx))
                im_lnm = SphSymbol('imln{lx}m{mx}'.format(lx=lx, mx=smx))

                EC += '''
                const REAL {re_lnm} = re_exp[{cx}];
                const REAL {im_lnm} = im_exp[{cx}];
                '''.format(
                    re_lnm=str(re_lnm),
                    im_lnm=str(im_lnm),
                    cx=str(cube_ind(lx, mx))
                )
                cm_re, cm_im = cmplx_mul(re_lnm, im_lnm, sph_gen.get_y_sym(lx, mx)[0],
                    sph_gen.get_y_sym(lx, mx)[1])
                EC += 'tmp_energy += ({cm_re}) * coeff;\n'.format(cm_re=cm_re)

                flops['*'] += 1
                flops['+'] += 1
                
            EC += 'rhol *= radius;\n'
            flops['*'] += 2

        header = str(Module(
            (
                Include('math.h'),
                Include('stdio.h'),
                Define('REAL', 'double'),
                Define('INT64', 'int64_t'),
                Define('NTERMS', L),
                Define('ESTRIDE', ncomp),
                Define('HESTRIDE', L**2),
                Define('CUBE_IND(L, M)', '((L) * ( (L) + 1 ) + (M) )'),
                Define('BLOCK_SIZE', 32)
            )
        ))

        LIB_PARAMETERS = """
                const INT64 num_movs,
                const INT64 * RESTRICT offsets,
                const REAL  * RESTRICT d_centres,
                const REAL  * RESTRICT d_positions,
                const REAL  * RESTRICT d_charges,
                const REAL  * RESTRICT d_local_exp,
                REAL * RESTRICT d_energy"""

        src = r"""
        {HEADER}
        
        extern "C" int indirect_interactions(
            {LIB_PARAMETERS}
        ){{
            
            const INT64 NBLOCKS = num_movs / BLOCK_SIZE;
            const INT64 BLOCK_END = NBLOCKS * BLOCK_SIZE;
            
            #pragma omp parallel for 
            for(INT64 bdx=0 ; bdx<NBLOCKS ; bdx++){{

                REAL radius_set[BLOCK_SIZE];
                //REAL phi_set[BLOCK_SIZE];
                REAL cos_theta_set[BLOCK_SIZE];
                REAL sqrt_theta_set[BLOCK_SIZE];
                REAL energy_set[BLOCK_SIZE];
                REAL sin_phi_set[BLOCK_SIZE];
                REAL cos_phi_set[BLOCK_SIZE];

                for(INT64 bix=0 ; bix<BLOCK_SIZE ; bix++){{
                    const INT64 idx = bdx*BLOCK_SIZE + bix;
                    const INT64 offset = offsets[idx];
                    const REAL dx = d_positions[idx*3 + 0] - d_centres[offset*3 + 0];
                    const REAL dy = d_positions[idx*3 + 1] - d_centres[offset*3 + 1];
                    const REAL dz = d_positions[idx*3 + 2] - d_centres[offset*3 + 2];

                    const REAL radius = sqrt(dx*dx + dy*dy + dz*dz);
                    const REAL phi = atan2(dy, dx);
                    const REAL theta = atan2(sqrt(dx*dx + dy*dy), dz);
                    
                    const REAL cos_theta = cos(theta);
                    const REAL sqrt_theta_tmp = sqrt(1.0 - cos_theta*cos_theta);

                    radius_set[bix] = radius;
                    //phi_set[bix]    = phi;
                    cos_theta_set[bix]  = cos(theta);
                    sqrt_theta_set[bix]  = sqrt_theta_tmp;
                    sin_phi_set[bix] = sin(phi);
                    cos_phi_set[bix] = cos(phi);
                }}
                
                #pragma omp simd simdlen(BLOCK_SIZE)
                for(INT64 bix=0 ; bix<BLOCK_SIZE ; bix++){{

                    const INT64 idx = bdx*BLOCK_SIZE + bix;
                    const INT64 offset = offsets[idx];

                    const REAL * RESTRICT re_exp = &d_local_exp[ESTRIDE * offset];
                    const REAL * RESTRICT im_exp = &d_local_exp[ESTRIDE * offset + HESTRIDE];                   
                    const REAL charge = d_charges[idx];
                    const REAL radius           = radius_set[bix];
                    //const REAL phi              = phi_set[bix];   
                    const REAL cos_theta        = cos_theta_set[bix];
                    const REAL sqrt_theta_tmp   = sqrt_theta_set[bix];
                    const REAL sin_phi          = sin_phi_set[bix];
                    const REAL cos_phi          = cos_phi_set[bix];

                    {SPH_GEN}

                    REAL tmp_energy = 0.0;
                    REAL rhol = 1.0;
                    REAL coeff = 0.0;

                    {ENERGY_COMP}

                    energy_set[bix] = tmp_energy;
                }}

                for(INT64 bix=0 ; bix<BLOCK_SIZE ; bix++){{
                    const INT64 idx = bdx*BLOCK_SIZE + bix;               
                    d_energy[idx] = energy_set[bix];
                }}

            }}

            #pragma omp parallel for schedule(static, 1)
            for(INT64 idx=BLOCK_END ; idx<num_movs ; idx++){{

                // map fmm cell into data structure
                const INT64 offset = offsets[idx];
                const REAL dx = d_positions[idx*3 + 0] - d_centres[offset*3 + 0];
                const REAL dy = d_positions[idx*3 + 1] - d_centres[offset*3 + 1];
                const REAL dz = d_positions[idx*3 + 2] - d_centres[offset*3 + 2];
                const REAL charge = d_charges[idx];

                const REAL * RESTRICT re_exp = &d_local_exp[ESTRIDE * offset];
                const REAL * RESTRICT im_exp = &d_local_exp[ESTRIDE * offset + HESTRIDE];
                
                const REAL radius = sqrt(dx*dx + dy*dy + dz*dz);
                const REAL phi = atan2(dy, dx);
                const REAL theta = atan2(sqrt(dx*dx + dy*dy), dz);
                const REAL cos_theta = cos(theta);
                const REAL sqrt_theta_tmp = sqrt(1.0 - cos_theta*cos_theta);

                const REAL sin_phi = sin(phi);
                const REAL cos_phi = cos(phi);

                //printf("C pos: %f %f %f centre %f %f %f \n", d_positions[idx*3 + 0], d_positions[idx*3 + 1], d_positions[idx*3 + 2], d_centres[offset*3 + 0], d_centres[offset*3 + 1], d_centres[offset*3 + 2]);
                //printf("C disp: %f %f %f\n", radius, theta, phi);

                {SPH_GEN}

                REAL tmp_energy = 0.0;
                REAL rhol = 1.0;
                REAL coeff = 0.0;

                {ENERGY_COMP}

                d_energy[idx] = tmp_energy;
            }}

            return 0;
        }}
        """.format(
            HEADER=header,
            LIB_PARAMETERS=LIB_PARAMETERS,
            SPH_GEN=str(sph_gen.module),
            ENERGY_COMP=EC
        )
        # print(flops, sum([flops[kx] for kx in flops.keys()]))
        return build.simple_lib_creator(header_code=' ', src_code=src)['indirect_interactions']



