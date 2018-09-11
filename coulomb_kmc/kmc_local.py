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
    def __init__(self, group, fmm, max_move):
        self.group = group
        self.fmm = fmm
        self.comm = fmm.tree.comm
        self.local_size = fmm.tree[-1].local_grid_cube_size
        self.local_offset = fmm.tree[-1].local_grid_offset
        
        self.entry_local_size = fmm.tree.entry_map.local_size
        self.entry_local_offset = fmm.tree.entry_map.local_offset


        ls = self.local_size
        els = self.entry_local_size

        self.cell_occupancy = np.zeros((ls[0], ls[1], ls[2], 1), dtype=INT64)
        self.entry_cell_occupancy = np.zeros((els[0], els[1], els[2], 1), dtype=INT64)
        self.entry_cell_occupancy_send = np.zeros((els[0], els[1], els[2], 1), dtype=INT64)

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

    
    def intiialise(self, positions, charges, fmm_cells, ids):

        self._cell_map = {}
        cell_occ = 1

        for pid in range(self.positions.npart_local):
            cell = self._get_fmm_cell(pid, slow_to_fast=True)
            if cell in self._cell_map.keys():
                self._cell_map[cell].append(pid)
                cell_occ = max(cell_occ, len(self._cell_map[cell]))
            else:
                self._cell_map[cell] = [pid]       
        

        elo = self.entry_local_offset
        els = self.entry_local_size

        self.comm.Barrier()
        self._win_ind.Fence(MPI.MODE_NOPUT)
        
        for cellx in product(
            range(elo[0], elo[0] + els[0]),
            range(elo[1], elo[1] + els[1]),
            range(elo[2], elo[2] + els[2])
            ):

            owning_rank = self.fmm.tree[-1].owners[cellx[0], cellx[1], cellx[2]]
            gcellx = self._global_cell_xyz((cellx[2], cellx[1], cellx[0]))
            self._win_ind.Get(self.entry_cell_occupancy[cellx[0], cellx[1], cellx[2], :],
                owning_rank, target=gcellx)
        
        self._win_ind.Fence(MPI.MODE_NOPUT)
        self.comm.Barrier()
        self._occ_win.Fence()
        
        for cellx in product(
            range(elo[0], elo[0] + els[0]),
            range(elo[1], elo[1] + els[1]),
            range(elo[2], elo[2] + els[2])
            ):

            particle_list = self._cell_map[cellx]
            num_particles = len(particle_list)

            owning_rank = self.fmm.tree[-1].owners[cellx[0], cellx[1], cellx[2]]
            owning_offset = self.entry_cell_occupancy[cellx[0], cellx[1], cellx[2], 0]

            self.entry_cell_occupancy_send[cellx[0], cellx[1], cellx[2], 0] = num_particles
            self._occ_win.Fetch_and_op(
                self.entry_cell_occupancy_send[cellx[0], cellx[1], cellx[2], :], # origin: this ranks n part
                self.entry_cell_occupancy[cellx[0], cellx[1], cellx[2], :],      # buffer for returned offset
                owning_rank,
                owning_offset
            )
        
        self._occ_win.Fence()
        self.comm.Barrier()



    def _get_fmm_cell(self, ix, slow_to_fast=False):
        R = self.fmm.R
        cc = self.group._fmm_cell[ix][0]
        sl = 2 ** (R - 1)
        cx = cc % sl
        cycz = (cc - cx) // sl
        cy = cycz % sl
        cz = (cycz - cy) // sl
        if not slow_to_fast:
            return cx, cy, cz
        else:
            return cz, cy, cx

    def _global_cell_xyz(self, tcx):
        """get global cell index from xyz tuple"""
        csc = self.fmm.tree.entry_map.cube_side_count
        gcs = [csc, csc, csc]
        return tcx[0] + gcs[0] * ( tcx[1] + gcs[1] * tcx[2] )





