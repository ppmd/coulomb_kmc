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
import time

from ppmd import mpi
from ppmd.coulomb.sph_harm import SphGen, SphSymbol, cmplx_mul, SphGenEphemeral
from ppmd.lib import build

from coulomb_kmc.common import BCType, PROFILE, spherical, cell_offsets, add_flop_dict
from coulomb_kmc.kmc_fmm_common import LocalOctalBase
from coulomb_kmc.kmc_expansion_tools import LocalExpEval

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
        
        self.cell_indices_arr = None
        self.flop_count_propose = None
        self.flop_count_accept = None

        self.cell_offsets = self.md.cell_offsets
        self.global_cell_size = self.md.global_cell_size
        self.entry_local_size = self.md.entry_local_size
        self.entry_local_offset = self.md.entry_local_offset
        self.periodic_factors = self.md.periodic_factors
        self.global_to_local = self.md.global_to_local
        self.boundary_condition = self.md.boundary_condition
        self._lee = LocalExpEval(self.fmm.L)

        local_store_dims = self.local_store_dims

        self.local_store_dims_arr = np.array(self.local_store_dims, dtype=INT64)

        self.local_expansions = np.zeros(
            local_store_dims + [2 * (self.fmm.L**2)], dtype=REAL)
        self.remote_inds = np.zeros(local_store_dims + [1], dtype=INT64)
        self.remote_inds[:] = -1
        

        self._ltp = self.fmm.tree_plain[-1]

        self._win = None
        self._win_ind = None
        
        self._ncomp = (self.fmm.L**2)*2

        # compute the cell centres
        self.cell_centres = np.zeros(local_store_dims + [3], dtype=REAL)
        self.orig_cell_centres = np.zeros_like(self.cell_centres)
        self._host_lib, self._flop_count_prop = self._init_host_kernels(self.fmm.L)
        self._host_accept_lib = self._init_accept_lib()
        self._init_host_point_eval()

    
    def _create_wins(self):
        assert self._win == None

        data_nbytes = self.fmm.tree_plain[-1][0,0,0,:].nbytes
        self._win = MPI.Win.Create(self._ltp, disp_unit=data_nbytes, comm=self.comm)

        assert self._win_ind == None
        self._win_ind = self.md.get_win_ind()


    def _free_wins(self):
        if self._win is not None:
            self._win.Free()
        self._win = None
        self.md.free_win_ind()
        self._win_ind = None


    def accept(self, movedata):
        """
        Accept a move using the coulomb_kmc internal accepted move data structure.

        :arg movedata: Move to accept.
        """

        self._accept(movedata, 0)
        #self._accept_py(movedata)


    def inject(self, movedata):
        """
        Inject a move using the coulomb_kmc internal accepted move data structure.

        :arg movedata: Move to accept.
        """
        self._accept(movedata, 1)


    def extract(self, movedata):
        """
        Inject a move using the coulomb_kmc internal accepted move data structure.

        :arg movedata: Move to accept.
        """
        self._accept(movedata, -1)


    def _accept(self, movedata, IE_FLAG=0):
        realdata = movedata[:7].view(dtype=REAL)

        old_position = realdata[0:3:]
        new_position = realdata[3:6:]
        charge       = realdata[6]
        gid          = movedata[7]
        old_fmm_cell = movedata[8]
        new_fmm_cell = movedata[9]
        
        # these are xyz
        new_cell_tuple = self._cell_lin_to_tuple_no_check(new_fmm_cell)
        old_cell_tuple = self._cell_lin_to_tuple_no_check(old_fmm_cell)

        lsd = self.local_store_dims
        
        exp_exec_count = INT64(0)
        
        t0 = time.time()
        self._accept_lib(
            INT64(IE_FLAG),
            REAL(charge),
            REAL(new_position[0]),
            REAL(new_position[1]),
            REAL(new_position[2]),
            REAL(old_position[0]),
            REAL(old_position[1]),
            REAL(old_position[2]),
            INT64(new_cell_tuple[0]),
            INT64(new_cell_tuple[1]),
            INT64(new_cell_tuple[2]),
            INT64(old_cell_tuple[0]),
            INT64(old_cell_tuple[1]),
            INT64(old_cell_tuple[2]),
            self.cell_indices_arr[0].ctypes.get_as_parameter(),
            self.cell_indices_arr[1].ctypes.get_as_parameter(),
            self.cell_indices_arr[2].ctypes.get_as_parameter(),
            INT64(lsd[0]),
            INT64(lsd[1]),
            INT64(lsd[2]),
            self.orig_cell_centres.ctypes.get_as_parameter(),
            self.local_expansions.ctypes.get_as_parameter(),
            ctypes.byref(exp_exec_count)
        )
        t1 = time.time()
        

        self._profile_inc('c_accept_exec_count_local_create', exp_exec_count.value)
        self._profile_inc('c_accept_time', t1 - t0)
        f = (
                self._profile_get('c_accept_flop_count_local_create') * \
                self._profile_get('c_accept_exec_count_local_create')
            ) / \
            self._profile_get('c_accept_time')
        self._profile_set('c_accept_gflop_rate', f/(10.**9))




    def _accept_py(self, movedata):
        assert self.comm.size == 1
        realdata = movedata[:7].view(dtype=REAL)

        old_position = realdata[0:3:]
        new_position = realdata[3:6:]
        charge       = realdata[6]
        gid          = movedata[7]
        old_fmm_cell = movedata[8]
        new_fmm_cell = movedata[9]

        new_cell_tuple = self._cell_lin_to_tuple_no_check(new_fmm_cell)
        old_cell_tuple = self._cell_lin_to_tuple_no_check(old_fmm_cell)
        
        old_tuple_s2f = tuple(reversed(old_cell_tuple))
        new_tuple_s2f = tuple(reversed(new_cell_tuple))

        lsd = self.local_store_dims

        if self.boundary_condition is BCType.FREE_SPACE:

            def old_cell_well_separated(cx):
                return not (
                    (abs(cx[0] - old_tuple_s2f[0]) < 2) and \
                    (abs(cx[1] - old_tuple_s2f[1]) < 2) and \
                    (abs(cx[2] - old_tuple_s2f[2]) < 2)
                )

            lsdi = (range(lsd[0]), range(lsd[1]), range(lsd[2]))

            for lsx in product(*lsdi):
                cellx = (self.cell_indices[0][lsx[0]], self.cell_indices[1][lsx[1]], 
                    self.cell_indices[2][lsx[2]])

                if old_cell_well_separated(cellx):# and \
                    #self.periodic_factors[0][lsx[0]] == 0 and \
                    #self.periodic_factors[1][lsx[1]] == 0 and \
                    #self.periodic_factors[2][lsx[2]] == 0:

                    cind = tuple(list(lsx) + [None])
                    cell_centre = self.orig_cell_centres[cind][0]
                    disp = spherical(tuple(old_position - cell_centre))
                    self._lee.local_exp(disp, -1.0 * charge, self.local_expansions[cind])

            def new_cell_well_separated(cx):
                return not (
                    (abs(cx[0] - new_tuple_s2f[0]) < 2) and \
                    (abs(cx[1] - new_tuple_s2f[1]) < 2) and \
                    (abs(cx[2] - new_tuple_s2f[2]) < 2)
                )

            for lsx in product(*lsdi):
                cellx = (self.cell_indices[0][lsx[0]], self.cell_indices[1][lsx[1]], 
                    self.cell_indices[2][lsx[2]])


                if new_cell_well_separated(cellx):# and \
                    #self.periodic_factors[0][lsx[0]] == 0 and \
                    #self.periodic_factors[1][lsx[1]] == 0 and \
                    #self.periodic_factors[2][lsx[2]] == 0:

                    cind = tuple(list(lsx) + [None])
                    cell_centre = self.orig_cell_centres[cind][0]
                    disp = spherical(tuple(new_position - cell_centre))

                    self._lee.local_exp(disp, charge, self.local_expansions[cind])

        elif self.boundary_condition in (BCType.PBC, BCType.NEAREST):

            R = self.fmm.R
            sl = 2 ** (R - 1)
            extent = self.domain.extent

            for ox in cell_offsets:

                fmm_cell_offset = (sl*ox[0], sl*ox[1], sl*ox[2])
                offset_pos = np.array((ox[2] * extent[0], ox[1] * extent[1], ox[0] * extent[2]))

                def old_cell_well_separated(cx):
                    return not (
                        (abs(cx[0] - old_tuple_s2f[0] - fmm_cell_offset[0]) < 2) and \
                        (abs(cx[1] - old_tuple_s2f[1] - fmm_cell_offset[1]) < 2) and \
                        (abs(cx[2] - old_tuple_s2f[2] - fmm_cell_offset[2]) < 2)
                    )

                
                lsdi = (range(lsd[0]), range(lsd[1]), range(lsd[2]))

                for lsx in product(*lsdi):
                    # get the original fmm cell in the primary image
                    cellx = (self.cell_indices[0][lsx[0]], self.cell_indices[1][lsx[1]], 
                        self.cell_indices[2][lsx[2]])
                    
                    # check if the original fmm cell was well separated
                    if old_cell_well_separated(cellx):

                        cind = tuple(list(lsx) + [None])
                        # original cell centre
                        cell_centre = self.orig_cell_centres[cind][0]
                        disp = spherical(tuple(old_position + offset_pos - cell_centre))
                        self._lee.local_exp(disp, -1.0 * charge, self.local_expansions[cind])

                def new_cell_well_separated(cx):
                    return not (
                        (abs(cx[0] - new_tuple_s2f[0] - fmm_cell_offset[0]) < 2) and \
                        (abs(cx[1] - new_tuple_s2f[1] - fmm_cell_offset[1]) < 2) and \
                        (abs(cx[2] - new_tuple_s2f[2] - fmm_cell_offset[2]) < 2)
                    )

                for lsx in product(*lsdi):
                    # get the original fmm cell in the primary image
                    cellx = (self.cell_indices[0][lsx[0]], self.cell_indices[1][lsx[1]], 
                        self.cell_indices[2][lsx[2]])
                    
                    # check if the original fmm cell was well separated
                    if new_cell_well_separated(cellx):

                        cind = tuple(list(lsx) + [None])
                        # original cell centre
                        cell_centre = self.orig_cell_centres[cind][0]
                        disp = spherical(tuple(new_position + offset_pos - cell_centre))
                        self._lee.local_exp(disp, charge, self.local_expansions[cind])

        else:
            raise NotImplementedError()




    def propose(self, total_movs, num_particles, host_data, cuda_data):
        """
        Propose a move using the coulomb_kmc internal proposed move data structures.
        For details see `coulomb_kmc.kmc_mpi_decomp.FMMMPIDecomp.setup_propose_with_dats`.
        """

        u0 = None
        u1 = None
        
        t0 = time.time()
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
            host_data['new_shifted_positions'].ctypes.get_as_parameter(),
            host_data['new_charges'].ctypes.get_as_parameter(),
            self.local_expansions.ctypes.get_as_parameter(),
            host_data['new_energy_i'].ctypes.get_as_parameter()
        )        
        t1 = time.time()

        u0 = host_data['old_energy_i'][:num_particles:]
        u1 = host_data['new_energy_i'][:total_movs:]

        nexec= num_particles + total_movs
        
        self._profile_inc('c_indirect_time', t1 - t0)
        self._profile_inc('c_indirect_exec_count', nexec)

        f = self._profile_get('c_indirect_exec_count') * self._flop_count_prop / \
            self._profile_get('c_indirect_time')
        
        self._profile_set('c_indirect_gflop_rate', f / (10.**9))

        return u0, u1


    def get_old_energy(self, num_particles, host_data):
        """
        Get old energies (for proposing extraction) using the coulomb_kmc internal proposed move data structures.
        For details see `coulomb_kmc.kmc_mpi_decomp.FMMMPIDecomp.setup_propose_with_dats`.
        """

        self._host_lib(
            INT64(num_particles),
            host_data['old_fmm_cells'].ctypes.get_as_parameter(),
            self.cell_centres.ctypes.get_as_parameter(),
            host_data['old_positions'].ctypes.get_as_parameter(),
            host_data['old_charges'].ctypes.get_as_parameter(),
            self.local_expansions.ctypes.get_as_parameter(),
            host_data['old_energy_i'].ctypes.get_as_parameter()
        )


    def initialise(self, positions, charges, fmm_cells):
        """
        Initialise the data structures for the indirect interactions.

        :arg positions: Initial positions of charges.
        :arg charges: Initial charge values.
        :arg fmm_cells: FMM cells of the input charges.
        """

        self.positions = positions
        self.charges = charges
        self.fmm_cells = fmm_cells
        self._get_local_expansions()
        self._compute_cell_centres()
        self.group = self.positions.group
        self.cell_indices_arr = [np.array(dimx, dtype=INT64) for dimx in self.cell_indices]

    def _compute_cell_centres(self):
        lsd = self.local_store_dims
        extent_s2f = list(reversed(self.domain.extent))
        ncell_s2f = self.global_cell_size
        widths = [ex/sx for ex, sx in zip(extent_s2f, ncell_s2f)]
        
        starts = [ -0.5*ex + 0.5*wx for wx, ex in zip(widths, extent_s2f)]
        offset_starts = [sx + wx*ox for sx, wx, ox in zip(starts, widths, self.md.local_offset)]
        
        centres = [
            [offset_starts[dimi] + cx * widths[dimi] for cx in dimx] for \
            dimi, dimx in enumerate(self.cell_offsets)
        ]

        orig_centres = [[starts[dimi] + cx * widths[dimi] for cx in dimx] for \
            dimi, dimx in enumerate(self.cell_indices)]


        # store the centres as xyz 
        for lcellx in product(range(lsd[0]), range(lsd[1]), range(lsd[2])):
            self.cell_centres[lcellx[0], lcellx[1], lcellx[2], :] = \
                (
                    centres[2][lcellx[2]],
                    centres[1][lcellx[1]], 
                    centres[0][lcellx[0]]
                )
            self.orig_cell_centres[lcellx[0], lcellx[1], lcellx[2], :] = \
                (
                    orig_centres[2][lcellx[2]],
                    orig_centres[1][lcellx[1]], 
                    orig_centres[0][lcellx[0]]
                )
    
    def _get_local_expansions(self):
        # copy/get the local expansions required from the fmm instance
        self._create_wins()
        rank = self.comm.rank
        ncomp = (self.fmm.L ** 2) * 2

        remote_cells = []
        
        self.comm.Barrier()
        
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
                
                self._win_ind.Lock(owning_rank, MPI.LOCK_SHARED)
                self._win_ind.Get(self.remote_inds[local_cellx[0], local_cellx[1], local_cellx[2], :],
                        owning_rank, target=gcellx)
                self._win_ind.Unlock(owning_rank)

                remote_cells.append((cellx, gcellx, owning_rank, local_cellx))

        self.comm.Barrier()

        # get the remote data
        for cell_tup in remote_cells:
            
            cellx = cell_tup[0]
            gcellx = cell_tup[1]
            owning_rank = cell_tup[2]
            local_cellx = cell_tup[3]
            
            remote_ind = self.remote_inds[local_cellx[0], local_cellx[1], local_cellx[2], 0]
            assert remote_ind > -1
            
            # TODO THIS IS NOT TECHNICALLY PORTABLE IN MPI STANDARD
            self._win.Lock(owning_rank, MPI.LOCK_SHARED)
            self._win.Get(
                    self.local_expansions[local_cellx[0], local_cellx[1], local_cellx[2], :],
                    owning_rank, target=remote_ind)
            self._win.Unlock(owning_rank)

        self.comm.Barrier()
        self._free_wins()
        self.md.free_win_ind()

    
    @staticmethod
    def _init_host_kernels(L):
        ncomp = (L**2)*2
        sph_gen = SphGen(maxl=L-1, theta_sym='theta', phi_sym='phi', ctype='double', avoid_calls=True, radius_symbol='coeff')
        
        flops = dict(sph_gen.flops)

        sph_gen = SphGenEphemeral(maxl=L-1, theta_sym='theta', phi_sym='phi', ctype='double', avoid_calls=True, radius_symbol='coeff')

        def cube_ind(l, m):
            return ((l) * ( (l) + 1 ) + (m) )

        
        radius_gen = 'const double {} = charge;\n'.format(sph_gen.get_radius_sym(0))
        for lx in range(1, L):
            radius_gen += 'const double {} = {} * radius;\n'.format(
                sph_gen.get_radius_sym(lx), sph_gen.get_radius_sym(lx-1))

        
        d = {}
        for lx in range(L):

            for mx in range(-lx, lx+1):
                smx = 'n' if mx < 0 else 'p'
                smx += str(abs(mx))

                re_lnm = SphSymbol('reln{lx}m{mx}'.format(lx=lx, mx=smx))
                im_lnm = SphSymbol('imln{lx}m{mx}'.format(lx=lx, mx=smx))

                EC = '''
                const REAL {re_lnm} = re_exp[{cx}];
                const REAL {im_lnm} = im_exp[{cx}];
                '''.format(
                    re_lnm=str(re_lnm),
                    im_lnm=str(im_lnm),
                    cx=str(cube_ind(lx, mx))
                )
                cm_re, cm_im = cmplx_mul(re_lnm, im_lnm, sph_gen.get_y_sym(lx, mx)[0],
                    sph_gen.get_y_sym(lx, mx)[1])
                EC += 'tmp_energy += {cm_re};\n'.format(cm_re=cm_re)

                d[(lx, mx)] = (EC,)

                flops['*'] += 1
                flops['+'] += 1
                
            flops['*'] += 2

        EC = sph_gen(d)

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
                Define('BLOCK_SIZE', 32),
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
            
            ///*
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

// hopefully the intel compiler will not generate broken code at some point
#ifndef __INTEL_COMPILER
#pragma omp simd simdlen(BLOCK_SIZE)
#endif
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

                    {RADIUS_GEN}
                    {SPH_GEN}

                    REAL tmp_energy = 0.0;

                    {ENERGY_COMP}

                    energy_set[bix] = tmp_energy;
                }}

                for(INT64 bix=0 ; bix<BLOCK_SIZE ; bix++){{
                    const INT64 idx = bdx*BLOCK_SIZE + bix;               
                    d_energy[idx] = energy_set[bix];
                }}

            }}
            //*/
            #pragma omp parallel for schedule(static, 1)
            //for(INT64 idx=0 ; idx<num_movs ; idx++){{
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

                //printf("-------------\n");

                //printf("rx %f, ry %f, rz %f | cx %f, cy %f, cz %f \n",
                //    d_positions[idx*3 + 0], d_positions[idx*3 + 1], d_positions[idx*3 + 2],
                //    d_centres[offset*3 + 0], d_centres[offset*3 + 1], d_centres[offset*3 + 2]
                //);
                //printf("dx %f, dy %f, dz %f | radius %f, theta %f, phi %f\n", dx, dy, dz, radius, theta, phi);
                //printf("re %f %f %f %f %f %f\n", re_exp[0], re_exp[1], re_exp[2], re_exp[3], re_exp[4], re_exp[5], re_exp[6], re_exp[7]);
                //printf("re %f %f %f %f %f %f\n", im_exp[0], im_exp[1], im_exp[2], im_exp[3], im_exp[4], im_exp[5], im_exp[6], im_exp[7]);
                //printf("-------------\n");

                const REAL sin_phi = sin(phi);
                const REAL cos_phi = cos(phi);
                
                {RADIUS_GEN}

                {SPH_GEN}

                REAL tmp_energy = 0.0;

                {ENERGY_COMP}

                d_energy[idx] = tmp_energy;
            }}

            return 0;
        }}
        """.format(
            HEADER=header,
            LIB_PARAMETERS=LIB_PARAMETERS,
            RADIUS_GEN=radius_gen,
            SPH_GEN='',
            ENERGY_COMP=EC
        )
        fc = sum([flops[kx] for kx in flops.keys()])
        _lib = build.simple_lib_creator(header_code=' ', src_code=src)['indirect_interactions'], fc

        
        return _lib


    def _init_accept_lib(self):
        L = self.fmm.L
        extent = self.fmm.domain.extent
        ncomp = (L**2)*2
        half_ncomp = (L**2)
        ncell_side = 2**(self.fmm.R - 1)
        
        sph_gen = SphGen(L-1, '_Y', 'theta', 'phi', radius_symbol='rhol')

        flops = sph_gen.flops.copy()

        # inject / extract cases
        if self.boundary_condition is BCType.FREE_SPACE:

            OFFSET_LOOPING_START = r'''

            for(INT64 ox=0 ; ox<1 ; ox++){
                const REAL mnew_posx = new_posx;
                const REAL mnew_posy = new_posy;
                const REAL mnew_posz = new_posz;

                const REAL mold_posx = old_posx;
                const REAL mold_posy = old_posy;
                const REAL mold_posz = old_posz;

                const INT64 mnew_tuplex = new_tuplex;
                const INT64 mnew_tupley = new_tupley;
                const INT64 mnew_tuplez = new_tuplez;

                const INT64 mold_tuplex = old_tuplex;
                const INT64 mold_tupley = old_tupley;
                const INT64 mold_tuplez = old_tuplez;
            '''

            OFFSET_LOOPING_START2 = r'''
            tcount += 1;
            for(INT64 ox=0 ; ox<1 ; ox++){
                        const REAL radius         = radius_vec[ox] ;
                        const REAL theta          = theta_vec[ox]  ;
                        const REAL phi            = phi_vec[ox]    ;
                        const bool well_separated_bool = bool_vec[ox]   ;       
            '''


        elif self.boundary_condition in (BCType.NEAREST, BCType.PBC):
            OFFSET_LOOPING_START = '''
            for(INT64 ox=0 ; ox<27 ; ox++){
                const INT64 o0 = OFFSETS[ox][0];
                const INT64 o1 = OFFSETS[ox][1];
                const INT64 o2 = OFFSETS[ox][2];

                const REAL mnew_posx = o2 * EXTENTX + new_posx;
                const REAL mnew_posy = o1 * EXTENTY + new_posy;
                const REAL mnew_posz = o0 * EXTENTZ + new_posz;

                const REAL mold_posx = o2 * EXTENTX + old_posx;
                const REAL mold_posy = o1 * EXTENTY + old_posy;
                const REAL mold_posz = o0 * EXTENTZ + old_posz;

                const INT64 mnew_tuplex = o2 * NCELL2 + new_tuplex;
                const INT64 mnew_tupley = o1 * NCELL1 + new_tupley;
                const INT64 mnew_tuplez = o0 * NCELL0 + new_tuplez;
                const INT64 mold_tuplex = o2 * NCELL2 + old_tuplex;
                const INT64 mold_tupley = o1 * NCELL1 + old_tupley;
                const INT64 mold_tuplez = o0 * NCELL0 + old_tuplez;
            '''

            OFFSET_LOOPING_START2 = r'''

            tcount += 27;
            for(INT64 ox=0 ; ox<27 ; ox++){
                        const REAL radius         = radius_vec[ox] ;
                        const REAL theta          = theta_vec[ox]  ;
                        const REAL phi            = phi_vec[ox]    ;
                        const bool well_separated_bool = bool_vec[ox]   ;       
            '''
            
            # in the hopping case these loops are called twice 
            flops['*'] += 12 * 27
            flops['+'] += 12 * 27

        else:
            raise NotImplementedError()

        OFFSET_LOOPING_END = r'''

                    radius_vec[ox] = radius;
                    theta_vec[ox] = theta;
                    phi_vec[ox] = phi;
                    bool_vec[ox] = well_separated_bool;

                }
        '''

        OFFSET_LOOPING_END2 = '}'



        # hopping case
        if self.boundary_condition is BCType.FREE_SPACE:


            HOP_OFFSET_LOOPING_START2 = r'''
            tcount += 2;
            for(INT64 ox=0 ; ox<2 ; ox++){
                        const REAL radius         = radius_vec[ox] ;
                        const REAL theta          = theta_vec[ox]  ;
                        const REAL phi            = phi_vec[ox]    ;
                        const REAL charge_loop         = charge_vec[ox]    ;
                        const bool well_separated_bool = bool_vec[ox];
            '''

            NEW_OFFSET = 1


        elif self.boundary_condition in (BCType.NEAREST, BCType.PBC):

            HOP_OFFSET_LOOPING_START2 = r'''

            tcount += 54;
            for(INT64 ox=0 ; ox<54 ; ox++){
                        const REAL radius         = radius_vec[ox] ;
                        const REAL theta          = theta_vec[ox]  ;
                        const REAL phi            = phi_vec[ox]    ;
                        const REAL charge_loop         = charge_vec[ox]    ;
                        const bool well_separated_bool = bool_vec[ox]   ;       
            '''

            NEW_OFFSET = 27
        else:
            raise NotImplementedError()

        HOP_OFFSET_LOOPING_END_OLD = r'''

                    radius_vec[ox] = radius;
                    theta_vec[ox] = theta;
                    phi_vec[ox] = phi;
                    charge_vec[ox] = -1.0 * charge;
                    bool_vec[ox] = well_separated_bool;

                }
        '''

        HOP_OFFSET_LOOPING_END_NEW = r'''

                    radius_vec[ox + {OFFSET}] = radius;
                    theta_vec[ox + {OFFSET}] = theta;
                    phi_vec[ox + {OFFSET}] = phi;
                    charge_vec[ox + {OFFSET}] = charge;
                    bool_vec[ox + {OFFSET}] = well_separated_bool;

                }}
        '''.format(
            OFFSET=NEW_OFFSET
        )


        HOP_OFFSET_LOOPING_END2 = '}'







        def cube_ind(L, M):
            return ((L) * ( (L) + 1 ) + (M) )



        # lib to create local expansions
        radius_gen = 'const double iradius = (well_separated_bool) ? 1.0/radius : 0.0;\n'
        radius_gen += 'const double {} = iradius * charge_loop;\n'.format(sph_gen.get_radius_sym(0))
        for lx in range(1, L):
            radius_gen += 'const double {} = {} * iradius;\n'.format(sph_gen.get_radius_sym(lx), sph_gen.get_radius_sym(lx-1))

        flops['*'] += L
        flops['/'] += 1


        assign_gen = ''
        for lx in range(L):
            for mx in range(-lx, lx+1):
                assign_gen += 'out[{ind}] += {ylmm};\n'.format(
                        ind=cube_ind(lx, mx),
                        ylmm=str(sph_gen.get_y_sym(lx, -mx)[0]),
                        l=lx
                    )
                assign_gen += 'out[{ind}] += {ylmm};\n'.format(
                        ind=cube_ind(lx, mx) + L*L,
                        ylmm=str(sph_gen.get_y_sym(lx, -mx)[1]),
                        l=lx
                    )

        flops['+'] +=  ncomp


        assign_header = str(sph_gen.header)


        header = str(Module(
            (
                Include('math.h'),
                Include('stdio.h'),
                Define('NCELL0', ncell_side),
                Define('NCELL1', ncell_side),
                Define('NCELL2', ncell_side),
                Define('EXTENTX', str(extent[0])),
                Define('EXTENTY', str(extent[1])),
                Define('EXTENTZ', str(extent[2])),
                Define('REAL', 'double'),
                Define('INT64', 'int64_t'),
                Define('NTERMS', L),
                Define('ESTRIDE', ncomp),
                Define('HESTRIDE', L**2),
                Initializer(Const(Value('INT64', 'OFFSETS[27][3]')),
                    '{' + ','.join([ '{' + ','.join([str(oxi) for oxi in ox]) + '}' \
                        for ox in cell_offsets ]) + '}'),
                Define('ABS(x)', '(((x) < 0) ? (-1*(x)) : (x))'),
                Define('IM_OFFSET', L*L),
            )
        )) + 3 * '\n' + assign_header

        
        sph_gen_v1 = str(sph_gen.module)
        assign_gen_v1 = str(assign_gen)


        to_test_gen = SphGenEphemeral(L-1, '_B', 'theta', 'phi', radius_symbol='rhol')
        d = {}
        for lx in range(L):
            for mx in range(-lx, lx+1):
                d[(lx, mx)] = (
                    'out[{ind}] += {ylmm};'.format(
                        ind=cube_ind(lx, mx),
                        ylmm=str(to_test_gen.get_y_sym(lx, -mx)[0]),
                        l=lx
                    ),
                    'out[{ind}] += {ylmm};'.format(
                        ind=cube_ind(lx, mx) + L*L,
                        ylmm=str(to_test_gen.get_y_sym(lx, -mx)[1]),
                        l=lx
                    )
                )

        sph_gen_v2 = to_test_gen(d)
        assign_gen_v2 = ''



        
        src = r'''
        {LOCAL_EXP_HEADER}

        {HEADER}
        
        static inline int well_separated(
            const INT64 t0,
            const INT64 t1,
            const INT64 t2,
            const INT64 f0,
            const INT64 f1,
            const INT64 f2
        ){{
            return ((ABS(t0-f0)<2) && (ABS(t1-f1)<2) && (ABS(t2-f2)<2)) ? 0 : 1;
        }}

        extern "C" int accept_local_exp(
            const INT64 INJECT_EXTRACT_FLAG,        // -1 => extract only, 0 i/e, 1 inject_only
            const REAL charge,
            const REAL new_posx,
            const REAL new_posy,
            const REAL new_posz,
            const REAL old_posx,
            const REAL old_posy,
            const REAL old_posz,

            const INT64 new_tuplex,
            const INT64 new_tupley,
            const INT64 new_tuplez,
            const INT64 old_tuplex,
            const INT64 old_tupley,
            const INT64 old_tuplez,

            const INT64 * RESTRICT cell_ind0, 
            const INT64 * RESTRICT cell_ind1, 
            const INT64 * RESTRICT cell_ind2, 
            const INT64 lsd0,
            const INT64 lsd1,
            const INT64 lsd2,

            const REAL * RESTRICT orig_centres,
            REAL * RESTRICT local_expansions,
            INT64 * RESTRICT local_exp_execute_count
        ){{
            INT64 tcount = 0;

            
            if ( INJECT_EXTRACT_FLAG == 0 ){{

                #pragma omp parallel for schedule(dynamic) collapse(3) reduction(+:tcount)
                for(INT64 cz=0 ; cz<lsd0 ; cz++){{
                    for(INT64 cy=0 ; cy<lsd1 ; cy++){{
                        for(INT64 cx=0 ; cx<lsd2 ; cx++){{

                            REAL out[ESTRIDE];

                            REAL radius_vec[54];
                            REAL theta_vec[54];
                            REAL phi_vec[54];
                            bool bool_vec[54];
                            REAL charge_vec[54];


                            const INT64 lin_cell = cx + lsd2*(cy + lsd1*cz);
                            const REAL centrex = orig_centres[lin_cell*3    ];
                            const REAL centrey = orig_centres[lin_cell*3 + 1];
                            const REAL centrez = orig_centres[lin_cell*3 + 2];
                            const INT64 fmm_cellx = cell_ind2[cx];
                            const INT64 fmm_celly = cell_ind1[cy];
                            const INT64 fmm_cellz = cell_ind0[cz];
                            REAL * RESTRICT cell_local_exp = &local_expansions[lin_cell*ESTRIDE];
                        
                            // zero the tmp vector
                            for (int tx=0 ; tx<ESTRIDE ; tx++) {{out[tx] = 0;}}

                            
                            {OFFSET_LOOPING_START}

                            const bool well_separated_bool = well_separated(mold_tuplex, mold_tupley, mold_tuplez, fmm_cellx, fmm_celly, fmm_cellz);

                            const REAL dx_old = mold_posx - centrex;
                            const REAL dy_old = mold_posy - centrey;
                            const REAL dz_old = mold_posz - centrez;
                            const REAL dx2_old = dx_old*dx_old;
                            const REAL dx2_p_dy2_old = dx2_old + dy_old*dy_old;
                            const REAL d2_old = dx2_p_dy2_old + dz_old*dz_old;
                            const REAL radius = sqrt(d2_old);
                            const REAL theta = atan2(sqrt(dx2_p_dy2_old), dz_old);
                            const REAL phi = atan2(dy_old, dx_old);

                            {HOP_OFFSET_LOOPING_END_OLD}


                            {OFFSET_LOOPING_START}

                            const bool well_separated_bool = well_separated(mnew_tuplex, mnew_tupley, mnew_tuplez, fmm_cellx, fmm_celly, fmm_cellz);

                            const REAL dx_new = mnew_posx - centrex;
                            const REAL dy_new = mnew_posy - centrey;
                            const REAL dz_new = mnew_posz - centrez;
                            const REAL dx2_new = dx_new*dx_new;
                            const REAL dx2_p_dy2_new = dx2_new + dy_new*dy_new;
                            const REAL d2_new = dx2_p_dy2_new + dz_new*dz_new;

                            const REAL radius = sqrt(d2_new);
                            const REAL theta = atan2(sqrt(dx2_p_dy2_new), dz_new);
                            const REAL phi = atan2(dy_new, dx_new);                       


                            {HOP_OFFSET_LOOPING_END_NEW}


                            {HOP_OFFSET_LOOPING_START2}


                            {RADIUS_GEN}
                            {SPH_GEN}
                            {ASSIGN_GEN}

                            {HOP_OFFSET_LOOPING_END2}


                            // copy the tmp vector into the global storage
                            for (int tx=0 ; tx<ESTRIDE ; tx++) {{cell_local_exp[tx] += out[tx];}}

                        }}
                    }}
                }}


            }} else if ( INJECT_EXTRACT_FLAG == 1 ){{

                #pragma omp parallel for schedule(dynamic) collapse(3) reduction(+:tcount)
                for(INT64 cz=0 ; cz<lsd0 ; cz++){{
                    for(INT64 cy=0 ; cy<lsd1 ; cy++){{
                        for(INT64 cx=0 ; cx<lsd2 ; cx++){{

                            REAL out[ESTRIDE];

                            REAL radius_vec[27];
                            REAL theta_vec[27];
                            REAL phi_vec[27];
                            bool bool_vec[27];


                            const INT64 lin_cell = cx + lsd2*(cy + lsd1*cz);
                            const REAL centrex = orig_centres[lin_cell*3    ];
                            const REAL centrey = orig_centres[lin_cell*3 + 1];
                            const REAL centrez = orig_centres[lin_cell*3 + 2];
                            const INT64 fmm_cellx = cell_ind2[cx];
                            const INT64 fmm_celly = cell_ind1[cy];
                            const INT64 fmm_cellz = cell_ind0[cz];
                            REAL * RESTRICT cell_local_exp = &local_expansions[lin_cell*ESTRIDE];
                        
                            // zero the tmp vector
                            for (int tx=0 ; tx<ESTRIDE ; tx++) {{out[tx] = 0;}}

                            
                            {OFFSET_LOOPING_START}

                            const bool well_separated_bool = well_separated(mnew_tuplex, mnew_tupley, mnew_tuplez, fmm_cellx, fmm_celly, fmm_cellz) && ( INJECT_EXTRACT_FLAG > -1 );

                            const REAL dx_new = mnew_posx - centrex;
                            const REAL dy_new = mnew_posy - centrey;
                            const REAL dz_new = mnew_posz - centrez;
                            const REAL dx2_new = dx_new*dx_new;
                            const REAL dx2_p_dy2_new = dx2_new + dy_new*dy_new;
                            const REAL d2_new = dx2_p_dy2_new + dz_new*dz_new;

                            const REAL radius = sqrt(d2_new);
                            const REAL theta = atan2(sqrt(dx2_p_dy2_new), dz_new);
                            const REAL phi = atan2(dy_new, dx_new);                       


                            {OFFSET_LOOPING_END}

                            {OFFSET_LOOPING_START2}

                            const REAL charge_loop = charge;

                            {RADIUS_GEN}
                            {SPH_GEN}
                            {ASSIGN_GEN}

                            {OFFSET_LOOPING_END2}


                            // copy the tmp vector into the global storage
                            for (int tx=0 ; tx<ESTRIDE ; tx++) {{cell_local_exp[tx] += out[tx];}}

                        }}
                    }}
                }}


            }} else {{

                #pragma omp parallel for schedule(dynamic) collapse(3) reduction(+:tcount)
                for(INT64 cz=0 ; cz<lsd0 ; cz++){{
                    for(INT64 cy=0 ; cy<lsd1 ; cy++){{
                        for(INT64 cx=0 ; cx<lsd2 ; cx++){{

                            REAL out[ESTRIDE];

                            REAL radius_vec[27];
                            REAL theta_vec[27];
                            REAL phi_vec[27];
                            bool bool_vec[27];


                            const INT64 lin_cell = cx + lsd2*(cy + lsd1*cz);
                            const REAL centrex = orig_centres[lin_cell*3    ];
                            const REAL centrey = orig_centres[lin_cell*3 + 1];
                            const REAL centrez = orig_centres[lin_cell*3 + 2];
                            const INT64 fmm_cellx = cell_ind2[cx];
                            const INT64 fmm_celly = cell_ind1[cy];
                            const INT64 fmm_cellz = cell_ind0[cz];
                            REAL * RESTRICT cell_local_exp = &local_expansions[lin_cell*ESTRIDE];
                        
                            // zero the tmp vector
                            for (int tx=0 ; tx<ESTRIDE ; tx++) {{out[tx] = 0;}}

                            
                            {OFFSET_LOOPING_START}
                            const bool well_separated_bool = well_separated(mold_tuplex, mold_tupley, mold_tuplez, fmm_cellx, fmm_celly, fmm_cellz) && ( INJECT_EXTRACT_FLAG < 1 );

                            const REAL dx_old = mold_posx - centrex;
                            const REAL dy_old = mold_posy - centrey;
                            const REAL dz_old = mold_posz - centrez;
                            const REAL dx2_old = dx_old*dx_old;
                            const REAL dx2_p_dy2_old = dx2_old + dy_old*dy_old;
                            const REAL d2_old = dx2_p_dy2_old + dz_old*dz_old;
                            const REAL radius = sqrt(d2_old);
                            const REAL theta = atan2(sqrt(dx2_p_dy2_old), dz_old);
                            const REAL phi = atan2(dy_old, dx_old);


                            {OFFSET_LOOPING_END}

                            {OFFSET_LOOPING_START2}

                            const REAL charge_loop= -1.0 * charge;

                            {RADIUS_GEN}
                            {SPH_GEN}
                            {ASSIGN_GEN}

                            {OFFSET_LOOPING_END2}



                            // copy the tmp vector into the global storage
                            for (int tx=0 ; tx<ESTRIDE ; tx++) {{cell_local_exp[tx] += out[tx];}}

                        }}
                    }}
                }}

            }}





            *local_exp_execute_count = tcount;
            return 0;
        }}
        '''.format(
            HEADER=str(header),
            OFFSET_LOOPING_START=OFFSET_LOOPING_START,
            OFFSET_LOOPING_END=OFFSET_LOOPING_END,
            OFFSET_LOOPING_START2=OFFSET_LOOPING_START2,
            OFFSET_LOOPING_END2=OFFSET_LOOPING_END2,            
            HOP_OFFSET_LOOPING_END_OLD=HOP_OFFSET_LOOPING_END_OLD,
            HOP_OFFSET_LOOPING_END_NEW=HOP_OFFSET_LOOPING_END_NEW,
            HOP_OFFSET_LOOPING_START2=HOP_OFFSET_LOOPING_START2,
            HOP_OFFSET_LOOPING_END2=HOP_OFFSET_LOOPING_END2,                       
            LOCAL_EXP_HEADER=self._lee.create_local_exp_header,
            LOCAL_EXP_SRC=self._lee.create_local_exp_src,
            SPH_GEN=str(sph_gen_v2),
            ASSIGN_GEN=str(assign_gen_v2),
            RADIUS_GEN=radius_gen
        )



        self.flop_count_accept = flops
        _t = self.flop_count_accept
        tf = sum([_tx for _tx in _t.values()])


        self._profile_inc('c_accept_flop_count_local_create', tf)
        self._accept_lib = build.simple_lib_creator(header_code=' ', src_code=src, name='octal_accept_lib')['accept_local_exp']
        
        


    def _init_host_point_eval(self):
        L = self.fmm.L

        ncomp = (L**2)*2
        sph_gen = SphGen(maxl=L-1, theta_sym='theta', phi_sym='phi', ctype='double', avoid_calls=True)
        
        def cube_ind(l, m):
            return ((l) * ( (l) + 1 ) + (m) )
        
        EC = ''
        for lx in range(L):
            EC += 'coeff = rhol;\n'

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
                Include('stdint.h'),
                Include('stdio.h'),
                Include('math.h'),
                Include('omp.h'),
                Define('REAL', 'double'),
                Define('INT64', 'int64_t'),
                Define('ESTRIDE', ncomp),
                Define('HESTRIDE', L**2),               
            )
        ))

        
        src = r"""
        static inline INT64 gcell_to_lcell(
            const INT64 * RESTRICT cell_data_offset,
            const INT64 * RESTRICT local_store_dims,
            const INT64 * cell
        ){{
            const INT64 c0 = cell[0] + cell_data_offset[2];
            const INT64 c1 = cell[1] + cell_data_offset[1];
            const INT64 c2 = cell[2] + cell_data_offset[0];
            return c0 + local_store_dims[2] * ( c1 + local_store_dims[1] * c2 );
        }}

        static inline void get_cell(
            const REAL * RESTRICT position,
            const REAL * RESTRICT extent,
            const INT64 * fmm_cells_per_side,
            INT64 * cell
        ){{

            REAL shifted_position[3];
            shifted_position[0] = position[0] + 0.5 * extent[0];
            shifted_position[1] = position[1] + 0.5 * extent[1];
            shifted_position[2] = position[2] + 0.5 * extent[2];

            const REAL w0 = fmm_cells_per_side[0] / extent[0];
            const REAL w1 = fmm_cells_per_side[1] / extent[1];
            const REAL w2 = fmm_cells_per_side[2] / extent[2];

            cell[0] = (INT64) (shifted_position[0] * w0);
            cell[1] = (INT64) (shifted_position[1] * w1);
            cell[2] = (INT64) (shifted_position[2] * w2);

            if (cell[0] >= fmm_cells_per_side[2]) {{ cell[0] = fmm_cells_per_side[2] - 1; }}
            if (cell[1] >= fmm_cells_per_side[1]) {{ cell[1] = fmm_cells_per_side[1] - 1; }}
            if (cell[2] >= fmm_cells_per_side[0]) {{ cell[2] = fmm_cells_per_side[0] - 1; }}

            return;
        }}




        extern "C" int indirect_point_eval(
            const INT64 N,
            const REAL  * RESTRICT d_positions,
            const REAL  * RESTRICT d_centres,
            const INT64 * RESTRICT cell_data_offset,
            const INT64 * RESTRICT local_store_dims,
            const INT64 * RESTRICT fmm_cells_per_side,
            const REAL  * RESTRICT extent,
            const REAL  * RESTRICT d_local_exp,
                  REAL  * RESTRICT d_potential
        ){{

            int err = 0;




#pragma omp parallel for
            for(INT64 idx=0 ; idx<N ; idx++){{

                INT64 ict[3];
                get_cell(&d_positions[idx*3], extent, fmm_cells_per_side, ict);
                const INT64 offset = gcell_to_lcell(cell_data_offset, local_store_dims, ict);

                const REAL dx = d_positions[idx*3 + 0] - d_centres[offset*3 + 0];
                const REAL dy = d_positions[idx*3 + 1] - d_centres[offset*3 + 1];
                const REAL dz = d_positions[idx*3 + 2] - d_centres[offset*3 + 2];

                const REAL * RESTRICT re_exp = &d_local_exp[ESTRIDE * offset];
                const REAL * RESTRICT im_exp = &d_local_exp[ESTRIDE * offset + HESTRIDE];
                
                const REAL radius = sqrt(dx*dx + dy*dy + dz*dz);
                const REAL phi = atan2(dy, dx);
                const REAL theta = atan2(sqrt(dx*dx + dy*dy), dz);
                const REAL cos_theta = cos(theta);
                const REAL sqrt_theta_tmp = sqrt(1.0 - cos_theta*cos_theta);

                const REAL sin_phi = sin(phi);
                const REAL cos_phi = cos(phi);

                {SPH_GEN}

                REAL tmp_energy = 0.0;
                REAL rhol = 1.0;
                REAL coeff = 0.0;

                {ENERGY_COMP}

                d_potential[idx] += tmp_energy;

            }}
            return err;
        }}

        """.format(
            SPH_GEN=str(sph_gen.module),
            ENERGY_COMP=EC
        )


        self._host_point_eval_lib = build.simple_lib_creator(
            header_code=header, src_code=src, name='kmc_fmm_indirect_point_eval')['indirect_point_eval']







    def eval_field(self, points, out):

        N = points.shape[0]
        ncps = (2**(self.fmm.R - 1))
        fmm_cells_per_side = np.array((ncps, ncps, ncps), dtype=INT64)
        extent = self.group.domain.extent
        e = np.zeros(3, REAL)
        e[:] = extent

        err = self._host_point_eval_lib(
            INT64(N),
            points.ctypes.get_as_parameter(),
            self.cell_centres.ctypes.get_as_parameter(),
            self.md.cell_data_offset.ctypes.get_as_parameter(),
            self.local_store_dims_arr.ctypes.get_as_parameter(),
            fmm_cells_per_side.ctypes.get_as_parameter(),
            e.ctypes.get_as_parameter(),
            self.local_expansions.ctypes.get_as_parameter(),
            out.ctypes.get_as_parameter()
        )

        assert err >= 0




