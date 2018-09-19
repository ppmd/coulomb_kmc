from __future__ import print_function, division

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"

import pytest, ctypes, math
from mpi4py import MPI
import numpy as np

np.set_printoptions(linewidth=200)


from ppmd import *
from ppmd.coulomb.fmm import *
from ppmd.coulomb.ewald_half import *
from scipy.special import sph_harm, lpmv
import time

from math import *
from itertools import product
MPISIZE = MPI.COMM_WORLD.Get_size()
MPIRANK = MPI.COMM_WORLD.Get_rank()
MPIBARRIER = MPI.COMM_WORLD.Barrier
DEBUG = True
SHARED_MEMORY = 'omp'

from coulomb_kmc import *

INT64 = ctypes.c_int64
REAL = ctypes.c_double

def test_fetch_op():
    
    a0 = np.zeros(1, dtype=INT64)
    a1 = np.zeros(1, dtype=INT64)
    b0 = np.zeros(1, dtype=INT64)
    b1 = np.zeros(MPISIZE, INT64)
    
    a0[:] = 1
    a1[:] = -1
    b1[:] = -1
    
    comm = MPI.COMM_WORLD
    gwin = MPI.Win()
    win = gwin.Create(
        b0,
        b0[0].nbytes,
        comm=comm
    )

    comm.Barrier()
    win.Fence()
    
    win.Fetch_and_op(a0, a1, 0, 0)

    win.Fence()
    comm.Barrier()

    comm.Gather(a1, b1)

    comm.Barrier()
    if MPIRANK == 0:
        assert b0[0] == MPISIZE
        required = set(range(MPISIZE))
        for bx in b1:
            required.remove(bx)
        assert len(required) == 0



def test_kmc_local_collection_1():
    # tests the collection of particles from all mpi ranks
    eps = 10.**-5
    L = 12
    ncomp = L*L*2
    R = 3

    N = 100
    E = 4.
    rc = E/4

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()
    A.npart = N

    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)
    A.ID = data.ParticleDat(ncomp=1, dtype=INT64)
    A.test_fmm_cell = data.ParticleDat(ncomp=1, dtype=INT64)
    A.PP = data.ParticleDat(ncomp=3)

    A.crr = data.ScalarArray(ncomp=1)

    rng = np.random.RandomState(seed=8657)
    
    
    ids = np.array(range(N))
    pos = np.zeros((N,3), dtype=REAL)
    chs = np.zeros((N,1), dtype=REAL)

    pos[:] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,3))
    for px in range(N):
        chs[px,0] = (-1.0)**(px+1)
    bias = np.sum(chs[:N:, 0])/N
    chs[:] -= bias


    A.P[:] = pos
    A.ID[:,0] = ids
    A.Q[:] = chs

    A.test_fmm_cell[:, 0] = 1

    A.scatter_data_from(0)
    
    # create a kmc instance
    kmc_fmm = KMCFMM(positions=A.P, charges=A.Q, 
       domain=A.domain, r=R, l=L, boundary_condition='free_space')
    # kmc_fmm.initialise()

    kmcl = kmc_local.LocalParticleData(A, kmc_fmm.fmm, 1.0)
    kmcl.initialise(A.P, A.Q, A.test_fmm_cell, A.ID)
    
    # assume the chosen  cell is on rank 0
    cids = kmcl._owner_store[0,0,1,:,4].view(dtype=INT64)
    pcs = kmcl._owner_store[0,0,1,:,:4].view(dtype=REAL)

    if MPIRANK == kmc_fmm.fmm.tree[-1].owners[0,0,1]:
        required = set(range(N))
        print("cids", cids)
        print("pcs", pcs)
        for pxi, px in enumerate(cids):
            assert pos[px, 0] == pcs[pxi, 0]
            assert pos[px, 1] == pcs[pxi, 1]
            assert pos[px, 2] == pcs[pxi, 2]
            assert chs[px, 0] == pcs[pxi, 3]
            assert ids[px] == px
            required.remove(px)
        assert len(required) == 0




def test_kmc_local_collection_2():
    # tests the redistribution of particle data post collection
    eps = 10.**-5
    L = 12
    ncomp = L*L*2
    R = 4
    E = 4.

    rng = np.random.RandomState(seed=8657)
    num_fmm_cells = 8**(R-1)
    ncells_per_side = 2**(R-1)
    
    def _cell_tup_to_lin(cx): return cx[2] + ncells_per_side * (cx[1] + ncells_per_side * cx[0])
    def _cell_to_extents(cx):
        l = -0.5 * E
        w = E/ncells_per_side
        keys = ('l0', 'l1', 'l2', 'h0', 'h1', 'h2')
        vals = [l + cxi*w for cxi in cx] + [l + (cxi+1)*w for cxi in cx]
        return dict(zip(keys, vals))

    # random occupancy
    
    cell_counts = np.zeros(num_fmm_cells, dtype=INT64)
    N = 0
    pos_store = {}
    for cellx in product(
            range(ncells_per_side), range(ncells_per_side), range(ncells_per_side)
        ):
        lin_cell = _cell_tup_to_lin(cellx)
        ncell = rng.randint(low=0, high=8)
        cell_counts[lin_cell] = ncell
        pos_store[cellx] = np.zeros((ncell, 5), dtype=REAL)
        cell_extents = _cell_to_extents(cellx)
        pos_store[cellx][:, 2] = rng.uniform(low=cell_extents['l0'], high=cell_extents['h0'], size=ncell)
        pos_store[cellx][:, 1] = rng.uniform(low=cell_extents['l1'], high=cell_extents['h1'], size=ncell)
        pos_store[cellx][:, 0] = rng.uniform(low=cell_extents['l2'], high=cell_extents['h2'], size=ncell)
        pos_store[cellx][:, 3] = rng.uniform(low=-2, high=2, size=ncell)
        pos_store[cellx][:, 4].view(dtype=INT64)[:] = np.arange(N, N+ncell)
        N += ncell

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()
    A.npart = N

    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)
    A.ID = data.ParticleDat(ncomp=1, dtype=INT64)
    A.test_fmm_cell = data.ParticleDat(ncomp=1, dtype=INT64)
    A.PP = data.ParticleDat(ncomp=3)

    A.crr = data.ScalarArray(ncomp=1)
    
    
    ids = np.zeros((N,1), dtype=INT64)
    pos = np.zeros((N,3), dtype=REAL)
    chs = np.zeros((N,1), dtype=REAL)
    
    tmp = 0
    for cellx in product(
            range(ncells_per_side), range(ncells_per_side), range(ncells_per_side)
        ):
        lin_cell = _cell_tup_to_lin(cellx)
        ncell = cell_counts[lin_cell]
        pos[tmp:tmp+ncell:, :] = pos_store[cellx][:,:3]
        chs[tmp:tmp+ncell:, 0] = pos_store[cellx][:,3]
        ids[tmp:tmp+ncell:, 0] = pos_store[cellx][:,4].view(dtype=INT64)
        A.test_fmm_cell[tmp:tmp+ncell:, 0] = lin_cell
        tmp += ncell


    A.P[:] = pos
    A.ID[:] = ids
    A.Q[:] = chs
    
    A.scatter_data_from(0)
    
    # create a kmc instance
    kmc_fmm = KMCFMM(positions=A.P, charges=A.Q, 
       domain=A.domain, r=R, l=L, boundary_condition='free_space')
    # kmc_fmm.initialise()

    kmcl = kmc_local.LocalParticleData(A, kmc_fmm.fmm, 1.0)
    
    
    nlocal = A.P.npart_local

    kmcl.initialise(A.P, A.Q, A.test_fmm_cell, A.ID)
    
    # loop over required cells and check the correct particle data exists
    
    for lcellx in product(
            range(kmcl.local_store_dims[0]),
            range(kmcl.local_store_dims[1]),
            range(kmcl.local_store_dims[2])
        ):
        gcellx = tuple([kmcl.cell_indices[dxi][dx] for dxi, dx in enumerate(lcellx)])
        lin_cell = _cell_tup_to_lin(gcellx)
        ncell = cell_counts[lin_cell]

        corr_data = pos_store[gcellx][:, :4]
        corr_ids = pos_store[gcellx][:, 4].view(dtype=INT64)
        
        kmcl_cell_data = kmcl.local_particle_store[lcellx[0], lcellx[1], lcellx[2], : ,:]

        kmcl_data = kmcl_cell_data[:, :4]
        kmcl_ids = kmcl_cell_data[:, 4].view(dtype=INT64)

        # check occupancy is correct
        assert kmcl.local_cell_occupancy[lcellx[0], lcellx[1], lcellx[2], 0] == ncell
    
        required = set(corr_ids)
        if ncell < 1:
            continue
        id_offset = corr_ids[0]
        for pxi in range(ncell):
            orig_id = kmcl_ids[pxi]-id_offset
            assert corr_ids[orig_id] == kmcl_ids[pxi]
            for compx in range(4):
                assert corr_data[orig_id, compx] == kmcl_data[pxi, compx]


            






