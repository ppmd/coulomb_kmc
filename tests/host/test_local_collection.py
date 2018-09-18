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

    N = 20
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










