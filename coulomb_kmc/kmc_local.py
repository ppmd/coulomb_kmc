from __future__ import division, print_function, absolute_import
__author__ = "W.R.Saunders"


import numpy as np
from math import ceil
import ctypes
from itertools import product
import os


REAL = ctypes.c_double
INT64 = ctypes.c_int64

# cuda imports if possible
import ppmd
import ppmd.cuda

from ppmd import mpi, runtime

MPI = mpi.MPI

if ppmd.cuda.CUDA_IMPORT:
    cudadrv = ppmd.cuda.cuda_runtime.cudadrv
    # the device should be initialised already by ppmd
    from pycuda.compiler import SourceModule
    import pycuda.gpuarray as gpuarray

# coulomb_kmc imports
from coulomb_kmc.common import BCType

_BUILD_DIR = runtime.BUILD_DIR

_offsets = (
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

class LocalParticleData(object):
    def __init__(self, fmm, max_move, boundary_condition=BCType.PBC, cuda=False):
        self.cuda_enabled = cuda
        
        if cuda and not ppmd.cuda.CUDA_IMPORT:
            print(ppmd.cuda.CUDA_IMPORT_ERROR)
            raise RuntimeError('CUDA was requested but failed to be initialised')

        self.fmm = fmm
        self.domain = fmm.domain
        self.comm = fmm.tree.cart_comm
        self.local_size = fmm.tree[-1].local_grid_cube_size
        self.local_offset = fmm.tree[-1].local_grid_offset
        
        self.entry_local_size = fmm.tree.entry_map.local_size
        self.entry_local_offset = fmm.tree.entry_map.local_offset
        
        assert boundary_condition in (BCType.PBC, BCType.FREE_SPACE, BCType.NEAREST)
        self._bc = boundary_condition

        ls = self.local_size
        lo = self.local_offset
        els = self.entry_local_size

        self.cell_occupancy = np.zeros((ls[0], ls[1], ls[2], 1), dtype=INT64)
        self.entry_cell_occupancy = np.zeros((els[0], els[1], els[2], 1), dtype=INT64)
        self.entry_cell_occupancy_send = np.zeros((els[0], els[1], els[2], 1), dtype=INT64)
        self.remote_inds = np.zeros((els[0], els[1], els[2], 1), dtype=INT64)

        self._wing = MPI.Win()
        self._occ_win = self._wing.Create(
            self.cell_occupancy,
            disp_unit=self.cell_occupancy[0,0,0,0].nbytes,
            comm=self.comm
        )
        gmap_nbytes = self.fmm.tree[-1].global_to_local[0,0,0].nbytes
        self._win_ind = self._wing.Create(
            self.fmm.tree[-1].global_to_local,
            disp_unit=gmap_nbytes,
            comm=self.comm
        )
        
        self._max_cell_occ = -1
        self._owner_store = None
        self.local_particle_store = None

        # compute the cells required for direct interactions

        csc = fmm.tree.entry_map.cube_side_count
        # in future domains may not be square

        # xyz tuple of domain dims
        csc = [csc, csc, csc]

        #slowest to fastest
        cscr = list(reversed(csc))

        csw = [self.domain.extent[0] / csc[0],
               self.domain.extent[1] / csc[1],
               self.domain.extent[2] / csc[2]]
        
        # this is pad per dimension slow to fast
        pad = [2 + int(ceil(max_move/cx)) for cx in reversed(csw)]
 
        # as offset indices slow to fast
        pad_low = [list(range(-px, 0)) for px in pad]
        pad_high = [list(range(lsx, lsx + px)) for px, lsx in zip(pad, ls)]
        
        # slowest to fastest to match octal tree indexing
        #global_to_local = [-lo[dx] + pad[dx] for dx in range(3)]
        #self.global_to_local = np.array(global_to_local, dtype=INT64)

        # cell indices as offsets from owned octal cells
        cell_indices = [ lpx + list(range(lsx)) + hpx for lpx, lsx, hpx in zip(pad_low, ls, pad_high) ]
        
        # periodic factors: slow to fast
        self.periodic_factors = [[ (lo[di] + cellx)//cscr[di] for cellx in dimx ] for di,dimx in enumerate(cell_indices)]
        
        # turn relative offsets into absolute cell indices
        cell_indices = [[ (cx + osx) % cscx for cx in dx ] for dx, cscx, osx in zip(cell_indices, cscr, lo)]
        
        # compute the offset to apply (addition) to fmm cells to map into the local store
        self.cell_data_offset = [len(px) - ox for px, ox in zip(pad_low, lo)]
        # this is now slowest to fastest not xyz
        # cell_indices = list(reversed(cell_indices))

        # this is slowest to fastest not xyz
        local_store_dims = [len(dx) for dx in cell_indices]
        
        # this is slowest to fastest not xyz
        self.local_store_dims = local_store_dims
        self.cell_indices = cell_indices

        self.remote_inds_particles = np.zeros((local_store_dims[0], local_store_dims[1], 
            local_store_dims[2], 1), dtype=INT64)

        self.local_cell_occupancy = np.zeros((local_store_dims[0], local_store_dims[1], 
            local_store_dims[2], 1), dtype=INT64)
        
        # force creation of self._owner_store and self.local_particle_store
        self._check_owner_store(max_cell_occ=1)

        self.positions = None
        self.charges = None
        self.fmm_cells = None
        self.ids = None
        self.group = None

        if self.cuda_enabled:
            # host copy of particle data for moves
            self._cuda_h = {}
            self._cuda_h['new_ids']           = np.zeros((1, 1), dtype=INT64)
            self._cuda_h['new_positions']     = np.zeros((1, 3), dtype=REAL)
            self._cuda_h['new_fmm_cells']     = np.zeros((1, 3), dtype=INT64)
            self._cuda_h['new_charges']       = np.zeros((1, 1), dtype=REAL)
            self._cuda_h['new_energy']        = np.zeros((1, 1), dtype=REAL)           
            self._cuda_h['old_positions']     = np.zeros((1, 3), dtype=REAL)
            self._cuda_h['old_fmm_cells']     = np.zeros((1, 3), dtype=INT64)
            self._cuda_h['old_charges']       = np.zeros((1, 1), dtype=REAL)
            self._cuda_h['old_energy']        = np.zeros((1, 1), dtype=REAL)
            self._cuda_h['old_ids']           = np.zeros((1, 1), dtype=INT64)
            # device copy of particle data for moves
            self._cuda_d = {}
            self._cuda_d['new_ids']       = None
            self._cuda_d['new_positions'] = None
            self._cuda_d['new_fmm_cells'] = None
            self._cuda_d['new_charges']   = None
            self._cuda_d['new_energy']    = None
            self._cuda_d['old_positions'] = None
            self._cuda_d['old_fmm_cells'] = None
            self._cuda_d['old_charges']   = None
            self._cuda_d['old_energy']    = None
            self._cuda_d['old_ids']       = None
            # device data for other particles
            # cell to particle map
            self._cuda_d_occupancy = None
            # particle data, is collected as [px, py, pz, chr, id] [REAL, REAL, REAL, REAL, INT64]
            self._cuda_d_pdata = None
            
            # pycuda funcs with kernels
            self._cuda_direct_new = None
            self._cuda_direct_old = None
            
            self._init_cuda_kernels()



    def propose(self, moves):
        total_movs = 0
        for movx in moves:
            movs = np.atleast_2d(movx[1])
            num_movs = movs.shape[0]
            total_movs += num_movs
        
        num_particles = len(moves)
        
        u0 = None
        u1 = None

        if self.cuda_enabled:
            self._resize_cuda_arrays(total_movs)
        
            tmp_index = 0

            for movi, movx in enumerate(moves):
                movs = np.atleast_2d(movx[1])
                num_movs = movs.shape[0]
                pid = movx[0]
                
                ts = tmp_index
                te = ts + num_movs

                self._cuda_h['new_positions'][ts:te:, :] = movs
                self._cuda_h['new_ids'][ts:te:, 0]       = self.ids[pid, 0]
                self._cuda_h['new_charges'][ts:te:, :]   = self.charges[pid, 0]

                self._cuda_h['old_positions'][movi, :] = self.positions[pid, :]
                self._cuda_h['old_charges'][movi, :]   = self.charges[pid, 0]
                self._cuda_h['old_fmm_cells'][movi, :] = self._get_fmm_cell(pid, self.fmm_cells)
                self._cuda_h['old_ids'][movi, 0]       = self.ids[pid, 0]

                for ti, tix in enumerate(range(tmp_index, tmp_index + num_movs)):
                    self._cuda_h['new_fmm_cells'][tix, :] = self._get_cell(movs[ti])
                tmp_index += num_movs

            self._copy_to_device()

            # test call to the direct_new func
            """
                        const INT64 d_num_movs,
                        const REAL  * d_positions,
                        const REAL  * d_charges,
                        const INT64 * d_ids,
                        const INT64 * d_fmm_cells,
                        const REAL  * d_pdata,
                        const INT64 * d_cell_occ,
                        const INT64 d_cell_stride,
                        REAL * d_energy
            """
            block_size = (256, 1, 1)
            grid_size = (int(ceil(total_movs/block_size[0])), 1)

            stride = self.local_particle_store.shape[3] * self.local_particle_store.shape[4]
            self._cuda_direct_new(
                np.int64(total_movs),
                self._cuda_d['new_positions'],
                self._cuda_d['new_charges'],
                self._cuda_d['new_ids'],
                self._cuda_d['new_fmm_cells'],
                self._cuda_d_pdata,
                self._cuda_d_occupancy,
                np.int64(stride),
                self._cuda_d['new_energy'],
                block=block_size,
                grid=grid_size
            )
            block_size = (256, 1, 1)
            grid_size = (int(ceil(num_particles/block_size[0])), 1)
            self._cuda_direct_old(
                np.int64(total_movs),
                self._cuda_d['old_positions'],
                self._cuda_d['old_charges'],
                self._cuda_d['old_ids'],
                self._cuda_d['old_fmm_cells'],
                self._cuda_d_pdata,
                self._cuda_d_occupancy,
                np.int64(stride),
                self._cuda_d['old_energy'],
                block=block_size,
                grid=grid_size
            )

            u1 = self._cuda_d['new_energy'].get()[:total_movs:, :]
            u0 = self._cuda_d['old_energy'].get()[:num_particles:, :]
        
        return u0, u1

    def _copy_to_device(self):
        assert self.cuda_enabled is True
        # copy the particle data to the device
        for keyx in self._cuda_h.keys():
            self._cuda_d[keyx] = gpuarray.to_gpu(self._cuda_h[keyx])
        # print(60*'-')
        # print(self._cuda_d_pdata.get())
        # print(60*'-')


    def _resize_cuda_arrays(self, total_movs):
        assert self.cuda_enabled is True
        if self._cuda_h['new_ids'].shape[0] < total_movs:
            for keyx in self._cuda_h.keys():
                ncomp = self._cuda_h[keyx].shape[1]
                dtype = self._cuda_h[keyx].dtype
                self._cuda_h[keyx] = np.zeros((total_movs, ncomp), dtype=dtype)


    def _check_owner_store(self, max_cell_occ):

        if self._owner_store is None or \
                max_cell_occ > self._owner_store.shape[3]:

            ls = self.local_size
            self._owner_store = np.zeros(
                (ls[0], ls[1], ls[2], max_cell_occ, 5), 
                dtype=REAL
            )
            nbytes = self._owner_store[0,0,0,0,:].nbytes
            self._win_global_store = self._wing.Create(
                self._owner_store,
                disp_unit=nbytes,
                comm=self.comm
            )
            # this stride is also used for the local store
            self._max_cell_occ = max_cell_occ
            
            lsd = self.local_store_dims
            self.local_particle_store = np.zeros(
                (lsd[0], lsd[1], lsd[2], max_cell_occ, 5), 
                dtype=REAL
            )

        else:
            self._owner_store.fill(0)

    
    def initialise(self, positions, charges, fmm_cells, ids):
        self.positions = positions
        self.charges = charges
        self.fmm_cells = fmm_cells
        self.ids = ids
        self.group = self.positions.group

        self._cell_map = {}
        cell_occ = 1

        for pid in range(positions.npart_local):
            cell = self._get_fmm_cell(pid, fmm_cells, slow_to_fast=True)
            
            
            if cell in self._cell_map.keys():
                self._cell_map[cell].append(pid)
                cell_occ = max(cell_occ, len(self._cell_map[cell]))
            else:
                self._cell_map[cell] = [pid]       
        
        lo = self.local_offset
        ls = self.local_size
        elo = self.entry_local_offset
        els = self.entry_local_size

        self.comm.Barrier()
        self._win_ind.Fence(MPI.MODE_NOPUT)
        
        for cellx in self._cell_map.keys():

            #cellx is a global index, lcellx is local
            lcellx = [cx - ox for cx, ox in zip(cellx, elo)]

            owning_rank = self.fmm.tree[-1].owners[cellx[0], cellx[1], cellx[2]]
            gcellx = self._global_cell_xyz((cellx[2], cellx[1], cellx[0]))
            self._win_ind.Get(self.remote_inds[lcellx[0], lcellx[1], lcellx[2], :],
                owning_rank, target=gcellx)
        
        for lcellx in product(
                range(self.local_store_dims[0]),
                range(self.local_store_dims[1]),
                range(self.local_store_dims[2])
            ):
            gcellx = [self.cell_indices[dxi][dx] for dxi, dx in enumerate(lcellx)]
            owning_rank = self.fmm.tree[-1].owners[gcellx[0], gcellx[1], gcellx[2]]
            gcellx = self._global_cell_xyz((gcellx[2], gcellx[1], gcellx[0]))


            if owning_rank != self.comm.rank:
                self._win_ind.Get(self.remote_inds_particles[lcellx[0], lcellx[1], lcellx[2], :],
                    owning_rank, target=gcellx)

        self._win_ind.Fence(MPI.MODE_NOPUT)
        self.comm.Barrier()
        self._occ_win.Fence()
        
        
        for cellx in self._cell_map.keys():

            #cellx is a global index, lcellx is local
            lcellx = [cx - ox for cx, ox in zip(cellx, elo)]
            
            particle_list = self._cell_map[cellx]
            
            num_particles = len(particle_list)

            owning_rank = self.fmm.tree[-1].owners[cellx[0], cellx[1], cellx[2]]
            owning_offset = self.remote_inds[lcellx[0], lcellx[1], lcellx[2], 0]

            self.entry_cell_occupancy_send[lcellx[0], lcellx[1], lcellx[2], 0] = num_particles
            self._occ_win.Fetch_and_op(
                self.entry_cell_occupancy_send[lcellx[0], lcellx[1], lcellx[2], :], # origin: this ranks npart
                self.entry_cell_occupancy[lcellx[0], lcellx[1], lcellx[2], :],      # buffer for returned offset
                owning_rank,
                owning_offset
            )

        self._occ_win.Fence()
        self.comm.Barrier()
        
        red_max_occ = np.array([np.max(self.cell_occupancy[:,:,:,0])], dtype=INT64)
        red_val = np.zeros_like(red_max_occ)
        self.comm.Allreduce(red_max_occ, red_val, MPI.MAX)
        self._check_owner_store(max_cell_occ=red_val[0])

        # entry_cell_offset should contain an offset this rank uses to send particle data
        
        self._win_global_store.Fence()

        tmp = []

        for cellx in self._cell_map.keys():

            owning_rank = self.fmm.tree[-1].owners[cellx[0], cellx[1], cellx[2]]
            owning_offset = self.entry_cell_occupancy[lcellx[0], lcellx[1], lcellx[2], 0]
            lcellx = [cx - ox for cx, ox in zip(cellx, elo)]
            
            particle_inds = np.array(self._cell_map[cellx])
            npart = particle_inds.shape[0]
            
            if owning_rank != self.comm.rank:
                # case for putting data
                tmp.append(np.zeros((npart, 5), dtype=REAL))
                # copy positons, charges, ids
                tmp[-1][:,0:3:] = positions[particle_inds, :]
                tmp[-1][:,3] = charges[particle_inds, 0]
                tmp[-1][:,4].view(dtype=INT64)[:] = ids[particle_inds, 0]
                
                # compute offset in remote buffer
                offset = self.remote_inds[lcellx[0], lcellx[1], lcellx[2], 0]
                offset *= self._max_cell_occ
                offset += self.entry_cell_occupancy[lcellx[0], lcellx[1], lcellx[2], 0]

                self._win_global_store.Put(tmp[-1], owning_rank, offset)
                

            else:
                # case for copying data directly

                s = self.entry_cell_occupancy[lcellx[0], lcellx[1], lcellx[2], 0]
                e = s + npart

                llcellx = [cx - ox for cx, ox in zip(cellx, lo)]
                self._owner_store[llcellx[0], llcellx[1], llcellx[2], s:e:, 0:3: ] = positions[particle_inds, :]
                self._owner_store[llcellx[0], llcellx[1], llcellx[2], s:e:, 3 ] = charges[particle_inds, 0]
                self._owner_store[llcellx[0], llcellx[1], llcellx[2], s:e:, 4 ].view(dtype=INT64)[:] = \
                    ids[particle_inds, 0]

        self._win_global_store.Fence()
        self.comm.Barrier()
        
        # at this point all ranks are holding the particle data for the octal cells they own

        # print(self._owner_store)
        
        # loop over required cells and copy particle data


        self._occ_win.Fence(MPI.MODE_NOPUT)
        self._win_global_store.Fence(MPI.MODE_NOPUT)

        self.local_particle_store[:] = -888


        for lcellx in product(
                range(self.local_store_dims[0]),
                range(self.local_store_dims[1]),
                range(self.local_store_dims[2])
            ):
            gcellx = [self.cell_indices[dxi][dx] for dxi, dx in enumerate(lcellx)]
            
            
            owning_rank = self.fmm.tree[-1].owners[gcellx[0], gcellx[1], gcellx[2]]
            if owning_rank == self.comm.rank:
                # do direct copy
                llcellx = [cx - ox for cx, ox in zip(gcellx, lo)]
                self.local_particle_store[lcellx[0], lcellx[1], lcellx[2], : , : ] = \
                        self._owner_store[llcellx[0], llcellx[1], llcellx[2], :, : ]

                self.local_cell_occupancy[lcellx[0], lcellx[1], lcellx[2], :] = \
                    self.cell_occupancy[llcellx[0], llcellx[1], llcellx[2], 0]

            else:
                # case to issue MPI_Get
                remote_index = self.remote_inds_particles[lcellx[0], lcellx[1], lcellx[2], 0]
                assert remote_index > -1

                self._occ_win.Get(self.local_cell_occupancy[lcellx[0], lcellx[1], lcellx[2], :],
                    owning_rank,
                    target=remote_index)

                remote_index *= self._max_cell_occ
                self._win_global_store.Get(
                    self.local_particle_store[lcellx[0], lcellx[1], lcellx[2], : , : ],
                    owning_rank,
                    target=remote_index
                )

        self._win_global_store.Fence(MPI.MODE_NOPUT)
        self._occ_win.Fence(MPI.MODE_NOPUT)
        self.comm.Barrier()
        

        # apply periodic boundary conditions
        for lcellx in product(
                range(self.local_store_dims[0]),
                range(self.local_store_dims[1]),
                range(self.local_store_dims[2])
            ):
            
            for dimx in range(3):
                dimx_xyz = 2 - dimx
                self.local_particle_store[lcellx[0], lcellx[1], lcellx[2], : , dimx_xyz:dimx_xyz+1: ] += \
                    self.periodic_factors[dimx][lcellx[dimx]] * self.domain.extent[dimx_xyz]

                if self._bc is BCType.FREE_SPACE and self.periodic_factors[dimx][lcellx[dimx]] != 0:
                    self.local_cell_occupancy[lcellx[0], lcellx[1], lcellx[2]] = 0
                    self.local_particle_store[lcellx[0], lcellx[1], lcellx[2], : , :] = np.nan


                elif self._bc is BCType.NEAREST and abs(self.periodic_factors[dimx][lcellx[dimx]]) > 1:
                    self.local_cell_occupancy[lcellx[0], lcellx[1], lcellx[2]] = 0
                    self.local_particle_store[lcellx[0], lcellx[1], lcellx[2], : , :] = np.nan

        # copy the particle data and the map to the device if applicable
        if self.cuda_enabled:
            self._cuda_d_occupancy = gpuarray.to_gpu(self.local_cell_occupancy)
            self._cuda_d_pdata = gpuarray.to_gpu(self.local_particle_store)



    def _get_fmm_cell(self, ix, cell_map, slow_to_fast=False):
        R = self.fmm.R
        cc = cell_map[ix][0]
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

        if not slow_to_fast:
            return cx, cy, cz
        else:
            return cz, cy, cx

    def _global_cell_xyz(self, tcx):
        """get global cell index from xyz tuple"""
        csc = self.fmm.tree.entry_map.cube_side_count
        gcs = [csc, csc, csc]
        return tcx[0] + gcs[0] * ( tcx[1] + gcs[1] * tcx[2] )

    def _get_cell(self, position):

        extent = self.group.domain.extent
        cell_widths = [ex / (2**(self.fmm.R - 1)) for ex in extent]
        spos = [0.5*ex + po for po, ex in zip(position, extent)]
        # if a charge is slightly out of the negative end of an axis this will
        # truncate to zero
        cell = [int(pcx / cwx) for pcx, cwx in zip(spos, cell_widths)]
        # truncate down if too high on axis, if way too high this should probably
        # throw an error
        return tuple([min(cx, 2**(self.fmm.R -1)) for cx in cell ])


    def _init_cuda_kernels(self):
        assert self.cuda_enabled
        
        assert REAL is ctypes.c_double # can be removed if the parameters are templated correctly
        #make the cuda kernel
        
        # precompute the cell offsets for the generated kernel
        # local_store_dims is slowest to fastest
        lsd = self.local_store_dims
        offsets = [str(ox[0] + lsd[2] * (ox[1] + lsd[1]*ox[2])) for ox in _offsets]
        offsets = '__device__ const INT32 OFFSETS[27] = {' + ','.join(offsets) + '};'
        
        offset_x = '__device__ const INT32 CELL_OFFSET_X = ' + str(self.cell_data_offset[2]) + ';'
        offset_y = '__device__ const INT32 CELL_OFFSET_Y = ' + str(self.cell_data_offset[1]) + ';'
        offset_z = '__device__ const INT32 CELL_OFFSET_Z = ' + str(self.cell_data_offset[0]) + ';'
        lsd_x = '__device__ const INT32 LSD_X = ' + str(self.local_store_dims[2]) + ';'
        lsd_y = '__device__ const INT32 LSD_Y = ' + str(self.local_store_dims[1]) + ';'
        lsd_z = '__device__ const INT32 LSD_Z = ' + str(self.local_store_dims[0]) + ';'

        common_1 = r"""
                const INT64 idx = threadIdx.x + blockIdx.x * blockDim.x;
                if (idx < d_num_movs){{
                    // this performs a cast but saves a register per value
                    // should never overflow as more than 2**3a1 cells per side is unlikely
                    // the offsets are slowest to fastest (numpy)
                    const INT32 icx = d_fmm_cells[idx*3]   + CELL_OFFSET_X;
                    const INT32 icy = d_fmm_cells[idx*3+1] + CELL_OFFSET_Y;
                    const INT32 icz = d_fmm_cells[idx*3+2] + CELL_OFFSET_Z;
                    
                    const INT32 ic = icx + LSD_X * (icy + LSD_Y*icz);
                    const REAL ipx = d_positions[idx*3];
                    const REAL ipy = d_positions[idx*3+1];
                    const REAL ipz = d_positions[idx*3+2];

                    REAL energy_red = 0.0;

                    // loop over the jcells
                    for(INT32 jcx=0 ; jcx<27 ; jcx++){{
                        const INT32 jc = ic + OFFSETS[jcx];

                        // compute the offset into the cell data
                        const INT32 offset = jc * ((INT32) d_cell_stride);

                        // loop over the particles in the j cell
                        for(INT32 jx=0 ; jx<d_cell_occ[jc] ; jx++){{            
                            const REAL jpx = d_pdata[offset + jx*5+0];
                            const REAL jpy = d_pdata[offset + jx*5+1];
                            const REAL jpz = d_pdata[offset + jx*5+2];
                            const REAL jch = d_pdata[offset + jx*5+3];
        """.format()

        common_2 = r"""
                        }}

                    }}
                    //printf("GPU: tmps %f, %f\n", energy_red, ich);

                    energy_red *= d_charges[idx];
                    d_energy[idx] = energy_red;

                }}
        """.format()
        
        KERNEL_PARAMETERS = """
                const INT64 d_num_movs,
                const REAL  * d_positions,
                const REAL  * d_charges,
                const INT64 * d_ids,
                const INT64 * d_fmm_cells,
                const REAL  * d_pdata,
                const INT64 * d_cell_occ,
                const INT64 d_cell_stride,
                REAL * d_energy
        """

        KERNEL_ARGS = "d_num_movs, d_positions, d_charges, d_ids, d_fmm_cells, d_pdata, d_cell_occ, d_cell_stride, d_energy"

        test_header = """
            #include <stdint.h>
            #include <stdio.h>
            #include <math.h>

            #define REAL double
            #define INT64 int64_t
            #define INT32 int32_t
        """

        src = r"""
            #include <stdint.h>
            #include <stdio.h>
            #include <math.h>

            #define REAL double
            #define INT64 int64_t
            #define INT32 int32_t

            {OFFSETS}
            {CELL_OFFSET_X}
            {CELL_OFFSET_Y}
            {CELL_OFFSET_Z}
            {LSD_X}
            {LSD_Y}
            //{LSD_Z}
            
            // only needed for the id which we don't
            // need to inspect for the new position


            __global__ void direct_new(
                {KERNEL_PARAMETERS}
            ) {{
                {COMMON_1}
                            energy_red += jch * rnorm3d(ipx - jpx, ipy - jpy, ipz - jpz);
                {COMMON_2}
            }}

            __global__ void direct_old(
                {KERNEL_PARAMETERS}
            ) {{
                {COMMON_1}
                            
                            const long long ll_jid =  __double_as_longlong(d_pdata[offset + jx*5+4]);
                            const int64_t jid = (int64_t) ll_jid;

                            // printf("\t\tGPU: jpos %f %f %f : jid %ld\n", jpx, jpy, jpz, jid);

                            if (jid != d_ids[idx]){{
                                energy_red += jch * rnorm3d(ipx - jpx, ipy - jpy, ipz - jpz);
                            }}
                {COMMON_2}
            }}


            """.format(
                REAL='double',
                INT64='int64_t',
                OFFSETS=offsets,
                CELL_OFFSET_X=offset_x,
                CELL_OFFSET_Y=offset_y,
                CELL_OFFSET_Z=offset_z,
                LSD_X=lsd_x,
                LSD_Y=lsd_y,
                LSD_Z=lsd_z,
                COMMON_1=common_1,
                COMMON_2=common_2,
                KERNEL_PARAMETERS=KERNEL_PARAMETERS
            )
        
        mod = SourceModule(src, options=['-O3','--use_fast_math'])
        self._cuda_direct_new = mod.get_function("direct_new")
        self._cuda_direct_old = mod.get_function("direct_old")
        # print("Num registers: new:", self._cuda_direct_new.num_regs, "old:", self._cuda_direct_old.num_regs)
        
        return
        # make a different wrapper to use the ppmd build system to avoid the pycuda build system

        LIB_PARAMETERS = """const int blockx,
        const int blocky,
        const int blockz,
        const int gridx,
        const int gridy,
        const int gridz,"""

        src += r"""
        extern "C" void wrapper_direct_new(
            {LIB_PARAMETERS} {KERNEL_PARAMETERS} ){{
            dim3 _B;
            dim3 _T;
            _B.x = blockx;
            _B.y = blocky;
            _B.z = blockz;

            direct_new<<<_B, _T>>>({KERNEL_ARGS});
            checkCudaErrors(cudaDeviceSynchronize());
        }}
        extern "C" void wrapper_direct_old(
            {LIB_PARAMETERS} {KERNEL_PARAMETERS} ){{
            dim3 _B;
            dim3 _T;
            _B.x = blockx;
            _B.y = blocky;
            _B.z = blockz;

            direct_old<<<_B, _T>>>({KERNEL_ARGS});
            checkCudaErrors(cudaDeviceSynchronize());
        }}

        """.format(
            KERNEL_PARAMETERS=KERNEL_PARAMETERS,
            KERNEL_ARGS=KERNEL_ARGS,
            LIB_PARAMETERS=LIB_PARAMETERS
        )

        print(src)
        
        src = """
        __global__ void foo(){
            const int a=1;
        }
        """
        test_header = """
        """
        self._cuda_lib = simple_lib_creator(header_code=test_header, src_code=src, name='kmc_cuda_local')







