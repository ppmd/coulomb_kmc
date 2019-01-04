from __future__ import print_function, division

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"

import pytest, ctypes, math
from mpi4py import MPI
import numpy as np
from itertools import product

np.set_printoptions(linewidth=200)


from ppmd import *
from ppmd.coulomb.fmm import *
from ppmd.coulomb.ewald_half import *
from scipy.special import sph_harm, lpmv
import time

from math import *

MPISIZE = MPI.COMM_WORLD.Get_size()
MPIRANK = MPI.COMM_WORLD.Get_rank()
MPIBARRIER = MPI.COMM_WORLD.Barrier
DEBUG = True
SHARED_MEMORY = 'omp'

from coulomb_kmc import *

REAL = ctypes.c_double
halfmeps = 0.5 - 10.0**-15



@pytest.mark.skipif('MPISIZE > 1')
def test_kmc_fmm_eval_field_1():
    """
    Tests that KMCFMM.eval_field agrees with a direct calculation
    """

    L = 12
    R = 3

    N = 50
    N2 = 10
    E = 4.
    rc = E/4

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()
    A.npart = N

    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)

    rng = np.random.RandomState(seed=1234)

    A.P[:] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,3))
    for px in range(N):
        A.Q[px,0] = (-1.0)**(px+1)
    bias = np.sum(A.Q[:N:, 0])/N
    A.Q[:, 0] -= bias
    
    A.scatter_data_from(0)
    
    bcs = 'free_space'
 
    kmc_fmm = KMCFMM(positions=A.P, charges=A.Q, 
        domain=A.domain, r=R, l=L, boundary_condition=bcs)
    kmc_fmm.initialise()


    eval_points = rng.uniform(low=-0.5*E, high=0.5*E, size=(N2, 3))

    correct_field = np.zeros(N2, dtype=REAL)
    for fx in range(N2):
        tmp = 0.0
        ptmp = eval_points[fx, :]
        for px in range(N):
            q = A.Q[px, 0]
            tmp += q / np.linalg.norm(ptmp - A.P[px, :])
        correct_field[fx] = tmp
    
    
    kmc_field = kmc_fmm.eval_field(eval_points)

    err = np.linalg.norm(correct_field - kmc_field, np.inf)
    assert err < 10.**-5



direction_bools = (
    (True, False, False),
    (False, True, False),
    (False, False, True)
)

@pytest.mark.parametrize("direction", direction_bools)
@pytest.mark.skipif('MPISIZE > 1')
def test_kmc_fmm_eval_field_2(direction):
    """
    Test the field is zero in the "middle" plane in the free space case.
    """

    L = 16
    R = 3

    N2 = 5
    E = 4.
    rc = E/4


    rng = np.random.RandomState(seed=562321)
    
    N = 20
    E = 4.0

    s = state.State()
    s.npart = N
    
    extent = [E/2 if bx else E for bx in direction]

    s.domain = domain.BaseDomainHalo(extent=extent)
    s.domain.boundary_condition = domain.BoundaryTypePeriodic()

    s.p = data.PositionDat()
    s.q = data.ParticleDat(ncomp=1)
    s.gid = data.ParticleDat(ncomp=1, dtype=ctypes.c_int64)

    for dimx in range(3):
        s.p[:N:, dimx] = rng.uniform(low=-halfmeps*extent[dimx], high=halfmeps*extent[dimx], size=(N))
    s.q[:] = rng.uniform(low=-2, high=2, size=(N, 1))
    s.gid[:, 0] = np.arange(0, N)

    mcs = kmc_dirichlet_boundary.MirrorChargeSystem(direction, s, 'p', 'q', 'gid')
    ms = mcs.mirror_state


    ms.scatter_data_from(0)
    
    bcs = 'free_space'
 
    kmc_fmm = KMCFMM(positions=ms.p, charges=ms.q, 
        domain=ms.domain, r=R, l=L, boundary_condition=bcs)
    kmc_fmm.initialise()

    
    if direction[0]: 
        plane_vector_1 = (0,1,0)
        plane_vector_2 = (0,0,1)
    elif direction[1]: 
        plane_vector_1 = (1,0,0)
        plane_vector_2 = (0,0,1)
    elif direction[2]: 
        plane_vector_1 = (1,0,0)
        plane_vector_2 = (0,1,0)
    else:
        raise RuntimeError('failed to set plane vectors')
    
    plane_vector_1 = np.array(plane_vector_1, dtype=REAL)
    plane_vector_2 = np.array(plane_vector_2, dtype=REAL)
    X,Y = np.meshgrid(
        np.linspace(-0.49999*E, 0.49999*E, N2),
        np.linspace(-0.49999*E, 0.49999*E, N2),
    )
    X = X.ravel()
    Y = Y.ravel()

    eval_points = np.zeros((N2*N2, 3), dtype=REAL)
    
    for px in range(N2*N2):
        eval_points[px, :] = X[px] * plane_vector_1 + Y[px] * plane_vector_2

    correct_field = np.zeros(eval_points.shape[0], dtype=REAL)
    for fx in range(eval_points.shape[0]):
        tmp = 0.0
        ptmp = eval_points[fx, :]
        for px in range(ms.npart):
            q = ms.q[px, 0]
            tmp += q / np.linalg.norm(ptmp - ms.p[px, :])
        correct_field[fx] = tmp
    
    kmc_field = kmc_fmm.eval_field(eval_points)

    err = np.linalg.norm(correct_field - kmc_field, np.inf)
    assert err < 10.**-5

    err = np.linalg.norm(kmc_field, np.inf)
    assert err < 10.**-5





