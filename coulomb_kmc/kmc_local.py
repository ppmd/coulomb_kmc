from __future__ import division, print_function, absolute_import
__author__ = "W.R.Saunders"


import numpy as np
from math import ceil
import ctypes
from itertools import product

from ppmd import mpi

MPI = mpi.MPI

REAL = ctypes.c_double
INT64 = ctypes.c_int64

class LocalParticleData(object):
    def __init__(self, fmm, max_move):
        self.fmm = fmm
        self.domain = fmm.domain
        self.comm = fmm.tree.cart_comm
        self.local_size = fmm.tree[-1].local_grid_cube_size
        self.local_offset = fmm.tree[-1].local_grid_offset
        
        self.entry_local_size = fmm.tree.entry_map.local_size
        self.entry_local_offset = fmm.tree.entry_map.local_offset


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
        csc = [csc, csc, csc]
        csw = [self.domain.extent[0] / csc[0],
               self.domain.extent[1] / csc[1],
               self.domain.extent[2] / csc[2]]
        
        # this is pad per dimension
        pad = [2 + int(ceil(max_move/cx)) for cx in csw]
 
        # as offset indices
        pad_low = [list(range(-px, 0)) for px in pad]
        pad_high = [list(range(lsx, lsx + px)) for px, lsx in zip(pad, reversed(ls))]
        
        # slowest to fastest to match octal tree indexing
        global_to_local = [-lo[dx] + pad[dx] for dx in reversed(range(3))]
        self.global_to_local = np.array(global_to_local, dtype=INT64)

        # cell indices as offsets from owned octal cells
        cell_indices = [ lpx + list(range(lsx)) + hpx for lpx, lsx, hpx in zip(pad_low, reversed(ls), pad_high) ]
        cell_indices = [[ (cx + osx) % cscx for cx in dx ] for dx, cscx, osx in zip(cell_indices, csc, reversed(lo))]

        # this is now slowest to fastest not xyz
        cell_indices = list(reversed(cell_indices))

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
    
        for lcellx in product(
                range(self.local_store_dims[0]),
                range(self.local_store_dims[1]),
                range(self.local_store_dims[2])
            ):
            gcellx = [self.cell_indices[dxi][dx] for dxi, dx in enumerate(lcellx)]
            
            owning_rank = self.fmm.tree[-1].owners[gcellx[0], gcellx[1], gcellx[2]]

            if self.local_cell_occupancy[lcellx[0], lcellx[1], lcellx[2], 0] > 0:
                #print(lcellx, gcellx)
                if owning_rank == self.comm.rank:
                    # print(self.local_particle_store[lcellx[0], lcellx[1], lcellx[2], : , : ])
                    pass
                else:
                    # print(self.local_particle_store[lcellx[0], lcellx[1], lcellx[2], : , : ])
                    pass


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





