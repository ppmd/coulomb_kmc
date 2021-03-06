from __future__ import print_function, division

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"

import pytest, ctypes, math
import numpy as np

np.set_printoptions(linewidth=200)


from ppmd import *
from ppmd.coulomb.fmm import *
from ppmd.coulomb.ewald_half import *
from scipy.special import sph_harm, lpmv
import time

from math import *
from itertools import product


from mpi4py import MPI
MPISIZE = MPI.COMM_WORLD.Get_size()
MPIRANK = MPI.COMM_WORLD.Get_rank()
MPIBARRIER = MPI.COMM_WORLD.Barrier
DEBUG = True
SHARED_MEMORY = 'omp'

from coulomb_kmc import *


def test_kmc_octal_1():
    eps = 10.**-5
    L = 12
    ncomp = L*L*2
    R = 4

    N = 20
    E = 4.
    rc = E/4

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()
    A.npart = N

    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)
    A.PP = data.ParticleDat(ncomp=3)

    A.crr = data.ScalarArray(ncomp=1)

    rng = np.random.RandomState(seed=8657)

    if N == 4:
        ra = 0.25 * E
        nra = -0.25 * E

        A.P[0,:] = ( 1.6,  1.6, 0.0)
        A.P[1,:] = (-1.500001,  1.499999, 0.0)
        A.P[2,:] = (-1.500001, -1.500001, 0.0)
        A.P[3,:] = ( 0.0,  0.0, 0.0)

        A.Q[0,0] = -1.
        A.Q[1,0] = 1.
        A.Q[2,0] = -1.
        A.Q[3,0] = 0.
    else:
        A.P[:] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,3))
        for px in range(N):
            A.Q[px,0] = (-1.0)**(px+1)
        bias = np.sum(A.Q[:N:, 0])/N
        A.Q[:, 0] -= bias

    A.scatter_data_from(0)
    
    # create a kmc instance
    kmc_fmm = KMCFMM(positions=A.P, charges=A.Q, 
        domain=A.domain, r=R, l=L, boundary_condition='free_space')
    # kmc_fmm.initialise()
    
    
    # make some data
    ns = kmc_fmm.fmm.tree[-1].ncubes_side_global
    test_data = np.array(rng.uniform(size=(ns, ns, ns, ncomp)), dtype=ctypes.c_double)
    
    ls = kmc_fmm.fmm.tree[-1].local_grid_cube_size
    lo = kmc_fmm.fmm.tree[-1].local_grid_offset

    # populate local part of octal tree
    kmc_fmm.fmm.tree_plain[-1][:,:,:,:] = test_data[
        lo[0]:lo[0]+ls[0]:,
        lo[1]:lo[1]+ls[1]:,
        lo[2]:lo[2]+ls[2]:,
        :
    ]
    
    mpi_decomp = kmc_mpi_decomp.FMMMPIDecomp(kmc_fmm.fmm, 1.0, common.BCType.FREE_SPACE)
    kmco = kmc_octal.LocalCellExpansions(mpi_decomp)
    kmco.initialise(positions=A.P, charges=A.Q, fmm_cells=None)

    lcl = kmco.cell_indices
    local_dims = kmco.local_store_dims
    for cellx in product(range(local_dims[0]), 
            range(local_dims[1]),
            range(local_dims[2])):

        orig_cell = (kmco.cell_indices[0][cellx[0]],
                kmco.cell_indices[1][cellx[1]],
                kmco.cell_indices[2][cellx[2]])

        np.testing.assert_array_equal(
            kmco.local_expansions[cellx[0], cellx[1], cellx[2], :],
            test_data[orig_cell[0], orig_cell[1], orig_cell[2], :]
        )


    kmc_fmm.free()

