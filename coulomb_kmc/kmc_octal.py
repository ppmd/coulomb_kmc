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
from coulomb_kmc.kmc_fmm_common import FMMMPIDecomp


MPI = mpi.MPI

REAL = ctypes.c_double
INT64 = ctypes.c_int64

class LocalCellExpansions(FMMMPIDecomp):
    """
    Object to get, store and update local expansions from an fmm instance.
    """
    def __init__(self, fmm, max_move, boundary_condition=BCType.PBC):

        self.fmm = fmm
        self.max_move = max_move
        self.domain = fmm.domain
        self.comm = fmm.tree.cart_comm

        csc = fmm.tree.entry_map.cube_side_count
        # in future domains may not be square, s2f
        csc = [csc, csc, csc]

        # s2f
        csw = [self.domain.extent[2] / csc[0],
               self.domain.extent[1] / csc[1],
               self.domain.extent[0] / csc[2]]
        
        # this is pad per dimension s2f
        pad = [1 + int(ceil(max_move/cx)) for cx in csw]
 
        ls = fmm.tree.entry_map.local_size
        lo = fmm.tree.entry_map.local_offset

        # as offset indices
        pad_low = [list(range(-px, 0)) for px in pad]
        pad_high = [list(range(lsx, lsx + px)) for px, lsx in zip(pad, ls)]
        
        # slowest to fastest to match octal tree indexing
        global_to_local = [-lo[dx] + pad[dx] for dx in range(3)]
        self.global_to_local = np.array(global_to_local, dtype=INT64)
        
        # cell indices as offsets from owned octal cells
        cell_indices = [ lpx + list(range(lsx)) + hpx for lpx, lsx, hpx in zip(pad_low, ls, pad_high) ]
 
        # s2f last allowable cell offset index
        self.upper_allowed = [cx[-2] for cx in cell_indices]
        self.lower_allowed = [cx[1] for cx in cell_indices]

        # use cell indices to get periodic coefficients
        self.periodic_factors = [[ (lo[di] + cellx)//csc[di] for cellx in dimx ] for \
            di, dimx in enumerate(cell_indices)]
        
        self.cell_offsets = cell_indices

        # cell indices as actual cell indices
        cell_indices = [[ (cx + osx) % cscx for cx in dx ] for dx, cscx, osx in zip(cell_indices, csc, lo)]

        # this is slowest to fastest (s2f) not xyz
        local_store_dims = [len(dx) for dx in cell_indices]
        
        cell_data_offset = [len(px) - ox for px, ox in zip(pad_low, lo)]
        self.cell_data_offset = np.array(cell_data_offset, dtype=INT64)

        # this is slowest to fastest not xyz
        self.local_store_dims = local_store_dims
        self.global_cell_size = csc
        self.cell_indices = cell_indices
        self.local_expansions = np.zeros(local_store_dims + [2 * (fmm.L**2)], dtype=REAL)
        self.remote_inds = np.zeros(local_store_dims + [1], dtype=INT64)
        self.remote_inds[:] = -1
        
        self._wing = MPI.Win()
        data_nbytes = self.fmm.tree_plain[-1][0,0,0,:].nbytes
        self._win = self._wing.Create(self.fmm.tree_plain[-1], disp_unit=data_nbytes, comm=self.comm)

        gmap_nbytes = self.fmm.tree[-1].global_to_local[0,0,0].nbytes
        self._win_ind = self._wing.Create(self.fmm.tree[-1].global_to_local, disp_unit=gmap_nbytes, comm=self.comm)
        
        self._ncomp = (self.fmm.L**2)*2

        # compute the cell centres
        self.cell_centres = np.zeros(local_store_dims + [3], dtype=REAL)


        # host copy of particle data for moves
        self._cuda_h = {}
        self._cuda_h['new_positions']     = np.zeros((1, 3), dtype=REAL)
        self._cuda_h['new_fmm_cells']     = np.zeros((1, 1), dtype=INT64)
        self._cuda_h['new_charges']       = np.zeros((1, 1), dtype=REAL)
        self._cuda_h['new_energy']        = np.zeros((1, 1), dtype=REAL)           
        self._cuda_h['old_positions']     = np.zeros((1, 3), dtype=REAL)
        self._cuda_h['old_fmm_cells']     = np.zeros((1, 1), dtype=INT64)
        self._cuda_h['old_charges']       = np.zeros((1, 1), dtype=REAL)
        self._cuda_h['old_energy']        = np.zeros((1, 1), dtype=REAL)
        
        self._host_lib = self._init_host_kernels(self.fmm.L)

    def propose(self):
        total_movs, num_particles = self._setup_propose(moves)
        
        u0 = None
        u1 = None
       
    def _resize_host_arrays(self, total_movs):
        if self._cuda_h['new_ids'].shape[0] < total_movs:
            for keyx in self._cuda_h.keys():
                ncomp = self._cuda_h[keyx].shape[1]
                dtype = self._cuda_h[keyx].dtype
                self._cuda_h[keyx] = np.zeros((total_movs, ncomp), dtype=dtype)

    def initialise(self):
        self._get_local_expansions()
        self._compute_cell_centres()

    def _compute_cell_centres(self):
        lsd = self.local_store_dims
        extent_s2f = list(reversed(self.domain.extent))
        ncell_s2f = self.global_cell_size
        widths = [ex/sx for ex, sx in zip(extent_s2f, ncell_s2f)]
        starts = [0.5*wx for wx in widths]
        centres = [[starts[dimi] + cx * widths[dimi] for cx in dimx] for dimi, dimx in enumerate(self.cell_offsets)]

        # store the centres as xyz 
        for lcellx in product(range(lsd[0]), range(lsd[1]), range(lsd[2])):
            self.cell_centres[lcellx[0], lcellx[1], lcellx[2], :] = \
                (centres[2][lcellx[2]], centres[1][lcellx[1]], centres[0][lcellx[0]])

    
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

        sph_gen = SphGen(maxl=L-1, theta_sym='theta', phi_sym='phi', ctype='double')
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
                

            EC += 'rhol *= radius;\n'


        header = str(Module(
            (
                Include('math.h'),
                Define('REAL', 'double'),
                Define('INT64', 'int64_t'),
                Define('NTERMS', L),
                Define('ESTRIDE', ncomp),
                Define('HESTRIDE', L**2),
                Define('CUBE_IND(L, M)', '((L) * ( (L) + 1 ) + (M) )')
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

        src = """
        {HEADER}
        
        extern "C" int indirect_interactions(
            {LIB_PARAMETERS}
        ){{
            #pragma omp parallel for simd simdlen(16)
            for(INT64 idx=0 ; idx<num_movs ; idx++){{
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

        return build.simple_lib_creator(header_code=' ', src_code=src)['indirect_interactions']



