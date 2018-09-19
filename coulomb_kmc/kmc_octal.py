from __future__ import division, print_function, absolute_import
__author__ = "W.R.Saunders"

"""
Octal Tree classes for kmc
"""

import numpy as np
from math import ceil
import ctypes
from itertools import product

from ppmd import mpi

MPI = mpi.MPI

REAL = ctypes.c_double
INT64 = ctypes.c_int64

class LocalCellExpansions(object):
    """
    Object to get, store and update local expansions from an fmm instance.
    """
    def __init__(self, fmm, max_move):
        self.fmm = fmm
        self.max_move = max_move
        self.domain = fmm.domain
        self.comm = fmm.tree.cart_comm

        csc = fmm.tree.entry_map.cube_side_count
        # in future domains may not be square
        csc = [csc, csc, csc]
        csw = [self.domain.extent[0] / csc[0],
               self.domain.extent[1] / csc[1],
               self.domain.extent[2] / csc[2]]
        
        # this is pad per dimension
        pad = [1 + int(ceil(max_move/cx)) for cx in csw]
 
        ls = fmm.tree.entry_map.local_size
        lo = fmm.tree.entry_map.local_offset

        # as offset indices
        pad_low = [list(range(-px, 0)) for px in pad]
        pad_high = [list(range(lsx, lsx + px)) for px, lsx in zip(pad, reversed(ls))]
        
        # slowest to fastest to match octal tree indexing
        global_to_local = [-lo[dx] + pad[dx] for dx in reversed(range(3))]
        self.global_to_local = np.array(global_to_local, dtype=INT64)

        # print("ls", ls, "lo", lo, "extent", self.domain.extent, "boundary", self.domain.boundary)
        
        # cell indices as offsets from owned octal cells
        cell_indices = [ lpx + list(range(lsx)) + hpx for lpx, lsx, hpx in zip(pad_low, reversed(ls), pad_high) ]
        cell_indices = [[ (cx + osx) % cscx for cx in dx ] for dx, cscx, osx in zip(cell_indices, csc, reversed(lo))]
        cell_indices = list(reversed(cell_indices))

        # this is slowest to fastest not xyz
        local_store_dims = [len(dx) for dx in cell_indices]

        
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
        

    def initialise(self):
        self._get_local_expansions()
    
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
        
    
    def _global_cell_xyz(self, tcx):
        """get global cell index from xyz tuple"""
        gcs = self.global_cell_size
        return tcx[0] + gcs[0] * ( tcx[1] + gcs[1] * tcx[2] )

















