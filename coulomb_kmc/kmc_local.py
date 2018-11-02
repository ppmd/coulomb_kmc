from __future__ import division, print_function, absolute_import
__author__ = "W.R.Saunders"

import numpy as np
from math import ceil
import ctypes
from itertools import product
import os
import time

from cgen import *
from cgen.cuda import *

REAL = ctypes.c_double
INT64 = ctypes.c_int64

# cuda imports if possible
import ppmd
import ppmd.cuda

from ppmd import mpi, runtime
from ppmd.lib import build


MPI = mpi.MPI

if ppmd.cuda.CUDA_IMPORT:
    cudadrv = ppmd.cuda.cuda_runtime.cudadrv
    # the device should be initialised already by ppmd
    from pycuda.compiler import SourceModule
    import pycuda.gpuarray as gpuarray

# coulomb_kmc imports
from coulomb_kmc.common import BCType, PROFILE
from coulomb_kmc.kmc_fmm_common import LocalOctalBase


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

class LocalParticleData(LocalOctalBase):

    def __init__(self, mpi_decomp):

        self.md = mpi_decomp
        self.comm = self.md.comm
        self.cuda_enabled = self.md.cuda_enabled
        self.fmm = self.md.fmm
        self.domain = self.md.domain
        self.local_store_dims = self.md.local_store_dims
        self.local_size = self.md.local_size
        self.local_offset = self.md.local_offset
        self.cell_indices = self.md.cell_indices
        self.cell_offsets = self.md.cell_offsets
        self.global_cell_size = self.md.global_cell_size
        self.entry_local_size = self.md.entry_local_size
        self.entry_local_offset = self.md.entry_local_offset
        self.periodic_factors = self.md.periodic_factors
        self.boundary_condition = self.md.boundary_condition

        ls = self.md.local_size
        lo = self.md.local_offset
        els = self.md.entry_local_size

        self.cell_occupancy = np.zeros((ls[0], ls[1], ls[2], 1), dtype=INT64)
        self.entry_cell_occupancy = np.zeros(
            (els[0], els[1], els[2], 1), dtype=INT64)
        self.entry_cell_occupancy_send = np.zeros(
                (els[0], els[1], els[2], 1), dtype=INT64)
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
        self.local_particle_store_ids = None
        
        lsd = self.local_store_dims

        self.local_store_dims_arr = np.array(lsd, dtype=INT64)
        self.remote_inds_particles = np.zeros((lsd[0], lsd[1], lsd[2], 1), dtype=INT64)
        self.local_cell_occupancy = np.zeros((lsd[0], lsd[1], lsd[2], 1), dtype=INT64)
        
        # force creation of self._owner_store and self.local_particle_store
        self._check_owner_store(max_cell_occ=1)

        self.positions = None
        self.charges = None
        self.fmm_cells = None
        self.ids = None
        self.group = None
        self._host_lib = None
        self._host_direct_new = None
        self._host_direct_old = None
        self._cuda_direct_new = None
        self._cuda_direct_old = None

        self.offsets_list = [str(ox[0] + self.local_store_dims[2] * (ox[1] + self.local_store_dims[1]*ox[2])) \
            for ox in _offsets]
        self.offsets_arr = np.array(self.offsets_list, dtype=INT64)
        
        if self.cuda_enabled:
            # device data for other particles
            # cell to particle map
            self._cuda_d_occupancy = None
            # particle data, is collected as [px, py, pz, chr, id] [REAL, REAL, REAL, REAL, INT64]
            self._cuda_d_pdata = None

            self._init_cuda_kernels()
        else:
            self._init_host_kernels()


    def accept(self, movedata):

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
        
        if self.boundary_condition is BCType.FREE_SPACE:
            old_locs = [[cxi for cxi, cx in enumerate(dims) if \
                ((cx == old_tuple_s2f[di]) and (abs(pbcs[cxi]) == 0))] for \
                di, (dims, pbcs) in enumerate(zip(self.cell_indices, self.periodic_factors))]

            new_locs = [[cxi for cxi, cx in enumerate(dims) if \
                ((cx == new_tuple_s2f[di]) and (abs(pbcs[cxi]) == 0))] for \
                di, (dims, pbcs) in enumerate(zip(self.cell_indices, self.periodic_factors))]

        elif self.boundary_condition in (BCType.PBC, BCType.NEAREST):
            old_locs = [[cxi for cxi, cx in enumerate(dims) if (cx == old_tuple_s2f[di])] for \
                di, dims in enumerate(self.cell_indices)]
            new_locs = [[cxi for cxi, cx in enumerate(dims) if (cx == new_tuple_s2f[di])] for \
                di, dims in enumerate(self.cell_indices)]
        else:
            raise RuntimeError("Bad/not implemented boundary condition")

        # add the new data if the new position is on this rank
        if (len(new_locs[0]) > 0) and (len(new_locs[1]) > 0) and (len(new_locs[2]) > 0):
            old_occupancy = self.local_cell_occupancy[new_locs[0][0], new_locs[1][0], new_locs[2][0], 0]
            possible_new_max = old_occupancy + 1
            # resize if needed
            self._resize_particle_store(possible_new_max)

            self.local_cell_occupancy[new_locs[0], new_locs[1], new_locs[2], 0] += 1
            self.local_particle_store_ids[new_locs[0], new_locs[1], new_locs[2], old_occupancy] = gid
            self.local_particle_store[new_locs[0], new_locs[1], new_locs[2], old_occupancy, 0] = new_position[0]
            self.local_particle_store[new_locs[0], new_locs[1], new_locs[2], old_occupancy, 1] = new_position[1]
            self.local_particle_store[new_locs[0], new_locs[1], new_locs[2], old_occupancy, 2] = new_position[2]
            self.local_particle_store[new_locs[0], new_locs[1], new_locs[2], old_occupancy, 3] = charge
            
            # insert the gid if cuda is used
            intview = self.local_particle_store[new_locs[0], new_locs[1], new_locs[2], old_occupancy, 4].view(
                dtype=INT64)
            intview[:] = gid

            # apply periodic boundary conditions to the newly inserted positions
            if self.boundary_condition in (BCType.PBC, BCType.NEAREST):
                extent = self.domain.extent
                for cellx in product(*new_locs):
                    xshift = self.periodic_factors[2][cellx[2]] * extent[0]
                    yshift = self.periodic_factors[1][cellx[1]] * extent[1]
                    zshift = self.periodic_factors[0][cellx[0]] * extent[2]
                    self.local_particle_store[cellx[0], cellx[1], cellx[2], old_occupancy, 0] += xshift
                    self.local_particle_store[cellx[0], cellx[1], cellx[2], old_occupancy, 1] += yshift
                    self.local_particle_store[cellx[0], cellx[1], cellx[2], old_occupancy, 2] += zshift


        # check this rank has relevevant cells for the old location
        if (len(old_locs[0]) > 0) and (len(old_locs[1]) > 0) and (len(old_locs[2]) > 0):
            # need to find the old location in the store
            old_occupancy = self.local_cell_occupancy[old_locs[0][0], old_locs[1][0], old_locs[2][0], 0]

            index = self.local_particle_store_ids[
                old_locs[0][0], old_locs[1][0], old_locs[2][0], :old_occupancy] == gid

            index = np.where(index)

            # this index should be unique, if not something else failed
            if len(index[0]) != 1 and old_fmm_cell != new_fmm_cell:
                print("Index was not unique, this is an error.")
                print(index)
                raise RuntimeError()
            elif len(index[0]) != 2 and old_fmm_cell == new_fmm_cell:
                raise RuntimeError()
            else:
                index = int(index[0][0])

            # set new occupancy
            self.local_cell_occupancy[old_locs[0], old_locs[1], old_locs[2], 0] -= 1
            
            # get the end data
            pos0 = self.local_particle_store[old_locs[0][0], old_locs[1][0], old_locs[2][0], old_occupancy-1, 0]
            pos1 = self.local_particle_store[old_locs[0][0], old_locs[1][0], old_locs[2][0], old_occupancy-1, 1]
            pos2 = self.local_particle_store[old_locs[0][0], old_locs[1][0], old_locs[2][0], old_occupancy-1, 2]
            char = self.local_particle_store[old_locs[0][0], old_locs[1][0], old_locs[2][0], old_occupancy-1, 3]
            gido = self.local_particle_store[old_locs[0][0], old_locs[1][0], old_locs[2][0], old_occupancy-1, 4]
            
            # shuffle the data down
            self.local_particle_store[old_locs[0], old_locs[1], old_locs[2], index, 0] = pos0
            self.local_particle_store[old_locs[0], old_locs[1], old_locs[2], index, 1] = pos1
            self.local_particle_store[old_locs[0], old_locs[1], old_locs[2], index, 2] = pos2
            self.local_particle_store[old_locs[0], old_locs[1], old_locs[2], index, 3] = char
            self.local_particle_store[old_locs[0], old_locs[1], old_locs[2], index, 4] = gido


    def propose(self, total_movs, num_particles, host_data, cuda_data):

        t0 = time.time()

        u0 = None
        u1 = None

        if self.cuda_enabled:

            block_size = (256, 1, 1)
            grid_size = (int(ceil(total_movs/block_size[0])), 1)
            stride = self.local_particle_store.shape[3] * self.local_particle_store.shape[4]
            self._cuda_direct_new(
                np.int64(total_movs),
                cuda_data['new_positions'],
                cuda_data['new_charges'],
                cuda_data['new_ids'],
                cuda_data['new_fmm_cells'],
                self._cuda_d_pdata,
                self._cuda_d_occupancy,
                np.int64(stride),
                cuda_data['new_energy_d'],
                block=block_size,
                grid=grid_size
            )
            block_size = (256, 1, 1)
            grid_size = (int(ceil(num_particles/block_size[0])), 1)
            self._cuda_direct_old(
                np.int64(num_particles),
                cuda_data['old_positions'],
                cuda_data['old_charges'],
                cuda_data['old_ids'],
                cuda_data['old_fmm_cells'],
                self._cuda_d_pdata,
                self._cuda_d_occupancy,
                np.int64(stride),
                cuda_data['old_energy_d'],
                block=block_size,
                grid=grid_size
            )

            u1 = cuda_data['new_energy_d'].get()[:total_movs:, :]
            u0 = cuda_data['old_energy_d'].get()[:num_particles:, :]
            self._profile_inc('cuda_direct', time.time() - t0)
        else:

            self._host_direct_new(
                INT64(total_movs),
                self.local_store_dims_arr.ctypes.get_as_parameter(),
                self.offsets_arr.ctypes.get_as_parameter(),
                host_data['new_positions'].ctypes.get_as_parameter(),
                host_data['new_charges'].ctypes.get_as_parameter(),
                host_data['new_ids'].ctypes.get_as_parameter(),
                host_data['new_fmm_cells'].ctypes.get_as_parameter(),
                self.local_particle_store.ctypes.get_as_parameter(),
                self.local_particle_store_ids.ctypes.get_as_parameter(),
                self.local_cell_occupancy.ctypes.get_as_parameter(),
                INT64(self.local_particle_store[0, 0, 0, :, 0].shape[0]),
                host_data['new_energy_d'].ctypes.get_as_parameter()
            )
            self._profile_inc('c_direct_new', time.time() - t0)
            t1 = time.time()
            self._host_direct_old(
                INT64(num_particles),
                self.local_store_dims_arr.ctypes.get_as_parameter(),
                self.offsets_arr.ctypes.get_as_parameter(),
                host_data['old_positions'].ctypes.get_as_parameter(),
                host_data['old_charges'].ctypes.get_as_parameter(),
                host_data['old_ids'].ctypes.get_as_parameter(),
                host_data['old_fmm_cells'].ctypes.get_as_parameter(),
                self.local_particle_store.ctypes.get_as_parameter(),
                self.local_particle_store_ids.ctypes.get_as_parameter(),
                self.local_cell_occupancy.ctypes.get_as_parameter(),
                INT64(self.local_particle_store[0, 0, 0, :, 0].shape[0]),
                host_data['old_energy_d'].ctypes.get_as_parameter()
            )
            self._profile_inc('c_direct_old', time.time() - t1)

            u1 = host_data['new_energy_d'][:total_movs:, :]
            u0 = host_data['old_energy_d'][:num_particles:, :]
        
        assert u1 is not None
        assert u0 is not None

        return u0, u1


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
            self.local_particle_store_ids = np.zeros(
                (lsd[0], lsd[1], lsd[2], max_cell_occ), 
                dtype=INT64
            )

        else:
            self._owner_store.fill(0)


    def _resize_particle_store(self, max_cell_occ):
        if self.local_particle_store.shape[3] < max_cell_occ:

            old_max = self.local_particle_store.shape[3]
            self._mac_cell_occ = max_cell_occ

            old_store = self.local_particle_store
            old_store_ids = self.local_particle_store_ids

            lsd = self.local_store_dims
            self.local_particle_store = np.zeros(
                (lsd[0], lsd[1], lsd[2], max_cell_occ, 5), 
                dtype=REAL
            )
            self.local_particle_store_ids = np.zeros(
                (lsd[0], lsd[1], lsd[2], max_cell_occ), 
                dtype=INT64
            )

            self.local_particle_store[:,:,:, :old_max, :] = old_store[:,:,:,:,:]
            self.local_particle_store_ids[:,:,:, :old_max] = old_store_ids[:,:,:,:]


    
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
        
        lo  = self.md.local_offset
        ls  = self.md.local_size
        elo = self.md.entry_local_offset
        els = self.md.entry_local_size

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

                if self.boundary_condition is BCType.FREE_SPACE and \
                        self.periodic_factors[dimx][lcellx[dimx]] != 0:
                    self.local_cell_occupancy[lcellx[0], lcellx[1], lcellx[2]] = 0
                    self.local_particle_store[lcellx[0], lcellx[1], lcellx[2], : , :] = np.nan


                elif self.boundary_condition is BCType.NEAREST and \
                        abs(self.periodic_factors[dimx][lcellx[dimx]]) > 1:
                    self.local_cell_occupancy[lcellx[0], lcellx[1], lcellx[2]] = 0
                    self.local_particle_store[lcellx[0], lcellx[1], lcellx[2], : , :] = np.nan

        # copy the particle data and the map to the device if applicable
        if self.cuda_enabled:
            self._cuda_d_occupancy = gpuarray.to_gpu(self.local_cell_occupancy)
            self._cuda_d_pdata = gpuarray.to_gpu(self.local_particle_store)
        else:
            self.local_particle_store_ids[:,:,:,:] = np.copy(self.local_particle_store[:,:,:,:,4].view(dtype=INT64))

    def _init_cuda_kernels(self):
        assert self.cuda_enabled
        
        assert REAL is ctypes.c_double # can be removed if the parameters are templated correctly
        #make the cuda kernel

        HEADER = str(Module(
            (
                Include('stdint.h'),
                Include('stdio.h'),
                Include('math.h'),
                Define('REAL', 'double'),
                Define('INT64', 'int64_t'),
                Define('INT32', 'int32_t'),
                Initializer(CudaDevice(Const(Value('INT32', 'OFFSETS[27]'))),
                    '{' + ','.join(self.offsets_list) + '}')
            )
        ))

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

        common_1 = r"""
                const INT64 idx = threadIdx.x + blockIdx.x * blockDim.x;
                if (idx < d_num_movs){{
                    // this performs a cast but saves a register per value
                    // should never overflow as more than 2**3a1 cells per side is unlikely
                    // the offsets are slowest to fastest (numpy)
                    const INT32 ic = d_fmm_cells[idx];
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
        # new/old part goes here ->
        common_2 = r"""
                        }}
                    }}
                    energy_red *= d_charges[idx];
                    d_energy[idx] = energy_red;
                }}
        """.format()
        

        src = r"""
            {HEADER}

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
                    if (jid != d_ids[idx]){{ energy_red += jch * rnorm3d(ipx - jpx, ipy - jpy, ipz - jpz); }}
                {COMMON_2}
            }}


            """.format(
                HEADER=HEADER,
                COMMON_1=common_1,
                COMMON_2=common_2,
                KERNEL_PARAMETERS=KERNEL_PARAMETERS
            )
        
        mod = SourceModule(src, options=['-O3','--use_fast_math'])
        self._cuda_direct_new = mod.get_function("direct_new")
        self._cuda_direct_old = mod.get_function("direct_old")
        
    def _init_host_kernels(self):

        header = str(Module(
            (
                Include('stdint.h'),
                Include('stdio.h'),
                Include('math.h'),
                Include('omp.h'),
                Define('REAL', 'double'),
                Define('INT64', 'int64_t')
            )
        ))

        LIB_PARAMETERS = """
                const INT64 num_movs,
                const INT64 * RESTRICT lsd,
                const INT64 * RESTRICT offsets,
                const REAL  * RESTRICT d_positions,
                const REAL  * RESTRICT d_charges,
                const INT64 * RESTRICT d_ids,
                const INT64 * RESTRICT d_fmm_cells,
                const REAL  * RESTRICT d_pdata,
                const INT64 * RESTRICT d_pdata_ids,
                const INT64 * RESTRICT d_cell_occ,
                const INT64 d_cell_stride,
                REAL * RESTRICT d_energy"""
            
        common_1 = r"""
                #pragma omp parallel for schedule(dynamic)
                for( INT64 idx=0 ; idx< num_movs ; idx++ ) {{

                    const INT64 ic = d_fmm_cells[idx];
                    const REAL ipx = d_positions[idx*3];
                    const REAL ipy = d_positions[idx*3+1];
                    const REAL ipz = d_positions[idx*3+2];

                    REAL energy_red = 0.0;

                    // loop over the jcells
                    for(INT64 jcx=0 ; jcx<27 ; jcx++){{
                        const INT64 jc = ic + offsets[jcx];

                        // compute the offset into the cell data
                        const INT64 offset = jc * d_cell_stride;
                        const INT64 offset5 = 5 * jc * d_cell_stride;

                        // loop over the particles in the j cell
                        for(INT64 jx=0 ; jx<d_cell_occ[jc] ; jx++){{            
                            const REAL jpx = d_pdata[offset5 + jx*5+0];
                            const REAL jpy = d_pdata[offset5 + jx*5+1];
                            const REAL jpz = d_pdata[offset5 + jx*5+2];
                            const REAL jch = d_pdata[offset5 + jx*5+3];
                            const REAL dx = ipx - jpx;
                            const REAL dy = ipy - jpy;
                            const REAL dz = ipz - jpz;
                            const REAL r2 = dx*dx + dy*dy + dz*dz;
                            const REAL contrib = jch / sqrt(r2);
        """.format()
        # new/old part goes here ->
        common_2 = r"""
                        }}
                    }}
                    energy_red *= d_charges[idx];
                    d_energy[idx] = energy_red;
                }}
            return 0;
        """.format()

        src = r"""
            {HEADER}

            extern "C" int direct_new(
                {LIB_PARAMETERS}
            ) {{
                {COMMON_1}
                            energy_red += contrib;
                {COMMON_2}
            }}

            extern "C" int direct_old(
                {LIB_PARAMETERS}
            ) {{
                {COMMON_1}
                            energy_red += (d_pdata_ids[offset + jx] != d_ids[idx]) ? contrib : 0.0;
                {COMMON_2}
            }}


            """.format(
                HEADER=header,
                COMMON_1=common_1,
                COMMON_2=common_2,
                LIB_PARAMETERS=LIB_PARAMETERS
            )
        
        self._host_lib = build.simple_lib_creator(header_code=' ', src_code=src, name='kmc_fmm_direct_host')
        self._host_direct_new = self._host_lib["direct_new"]
        self._host_direct_old = self._host_lib["direct_old"]







