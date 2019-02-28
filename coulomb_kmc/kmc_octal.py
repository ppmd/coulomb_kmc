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
from ppmd.coulomb.sph_harm import SphGen, SphSymbol, cmplx_mul
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
        self.orig_cell_centres = np.zeros_like(self.cell_centres)
        self._host_lib, self._flop_count_prop = self._init_host_kernels(self.fmm.L)
        self._host_accept_lib = self._init_accept_lib()




    def accept(self, movedata):
        self._accept(movedata)
        #self._accept_py(movedata)

    def _accept(self, movedata):
        realdata = movedata[:7].view(dtype=REAL)

        old_position = realdata[0:3:]
        new_position = realdata[3:6:]
        charge       = realdata[6]
        gid          = movedata[7]
        old_fmm_cell = movedata[8]
        new_fmm_cell = movedata[9]
        
        # these are xyz
        new_cell_tuple = self._cell_lin_to_tuple(new_fmm_cell)
        old_cell_tuple = self._cell_lin_to_tuple(old_fmm_cell)

        lsd = self.local_store_dims
        
        exp_exec_count = INT64(0)
        
        t0 = time.time()
        self._accept_lib(
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


    def initialise(self, positions, charges, fmm_cells):
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

                printf("-------------\n");

                printf("rx %f, ry %f, rz %f | cx %f, cy %f, cz %f \n",
                    d_positions[idx*3 + 0], d_positions[idx*3 + 1], d_positions[idx*3 + 2],
                    d_centres[offset*3 + 0], d_centres[offset*3 + 1], d_centres[offset*3 + 2]
                );
                printf("dx %f, dy %f, dz %f | radius %f, theta %f, phi %f\n", dx, dy, dz, radius, theta, phi);
                printf("re %f %f %f %f %f %f\n", re_exp[0], re_exp[1], re_exp[2], re_exp[3], re_exp[4], re_exp[5], re_exp[6], re_exp[7]);
                printf("re %f %f %f %f %f %f\n", im_exp[0], im_exp[1], im_exp[2], im_exp[3], im_exp[4], im_exp[5], im_exp[6], im_exp[7]);
                printf("-------------\n");

                const REAL sin_phi = sin(phi);
                const REAL cos_phi = cos(phi);

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
        fc = sum([flops[kx] for kx in flops.keys()])
        return build.simple_lib_creator(header_code=' ', src_code=src)['indirect_interactions'], fc


    def _init_accept_lib(self):
        L = self.fmm.L
        extent = self.fmm.domain.extent
        ncomp = (L**2)*2
        half_ncomp = (L**2)
        ncell_side = 2**(self.fmm.R - 1)
        
        if self.boundary_condition is BCType.FREE_SPACE:
            OFFSET_LOOPING_START = r'''
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
            OFFSET_LOOPING_END = r'''
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

            OFFSET_LOOPING_END = '}'
        else:
            raise NotImplementedError()


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
            )
        ))
        
        src = r'''
        {LOCAL_EXP_HEADER}
        {LOCAL_EXP_SRC}

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
            #pragma omp parallel for schedule(dynamic) collapse(3) reduction(+:tcount)
            for(INT64 cz=0 ; cz<lsd0 ; cz++){{
                for(INT64 cy=0 ; cy<lsd1 ; cy++){{
                    for(INT64 cx=0 ; cx<lsd2 ; cx++){{
                        const INT64 lin_cell = cx + lsd2*(cy + lsd1*cz);
                        const REAL centrex = orig_centres[lin_cell*3    ];
                        const REAL centrey = orig_centres[lin_cell*3 + 1];
                        const REAL centrez = orig_centres[lin_cell*3 + 2];
                        const INT64 fmm_cellx = cell_ind2[cx];
                        const INT64 fmm_celly = cell_ind1[cy];
                        const INT64 fmm_cellz = cell_ind0[cz];
                        
                        {OFFSET_LOOPING_START}

                        REAL * RESTRICT cell_local_exp = &local_expansions[lin_cell*ESTRIDE];

                        if (well_separated(mold_tuplex, mold_tupley, mold_tuplez, fmm_cellx, fmm_celly, fmm_cellz)){{
                            const REAL dx = mold_posx - centrex;
                            const REAL dy = mold_posy - centrey;
                            const REAL dz = mold_posz - centrez;
                            const REAL dx2 = dx*dx;
                            const REAL dx2_p_dy2 = dx2 + dy*dy;
                            const REAL d2 = dx2_p_dy2 + dz*dz;
                            const REAL radius = sqrt(d2);
                            const REAL theta = atan2(sqrt(dx2_p_dy2), dz);
                            const REAL phi = atan2(dy, dx);
                            inline_local_exp(-1.0 * charge, radius, theta, phi, cell_local_exp);

                            tcount++;
                        }}

                        if (well_separated(mnew_tuplex, mnew_tupley, mnew_tuplez, fmm_cellx, fmm_celly, fmm_cellz)){{
                            const REAL dx = mnew_posx - centrex;
                            const REAL dy = mnew_posy - centrey;
                            const REAL dz = mnew_posz - centrez;
                            const REAL dx2 = dx*dx;
                            const REAL dx2_p_dy2 = dx2 + dy*dy;
                            const REAL d2 = dx2_p_dy2 + dz*dz;
                            const REAL radius = sqrt(d2);
                            const REAL theta = atan2(sqrt(dx2_p_dy2), dz);
                            const REAL phi = atan2(dy, dx);
                            inline_local_exp(charge, radius, theta, phi, cell_local_exp);

                            tcount++;
                        }}

                        {OFFSET_LOOPING_END}

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
            LOCAL_EXP_HEADER=self._lee.create_local_exp_header,
            LOCAL_EXP_SRC=self._lee.create_local_exp_src
        )
        self.flop_count_accept = self._lee.flop_count_create_local_exp
        _t = self.flop_count_accept
        tf = sum([_tx for _tx in _t.values()])
        self._profile_inc('c_accept_flop_count_local_create', tf)
        self._accept_lib = build.simple_lib_creator(header_code=' ', src_code=src)['accept_local_exp']























