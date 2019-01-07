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





direction_bools = (
    (True, False, False),
    (False, True, False),
    (False, False, True)
)

@pytest.mark.parametrize("direction", direction_bools)
@pytest.mark.skipif('MPISIZE > 1')
def test_kmc_fmm_eval_field_3(direction):
    """
    Test the field is zero in the "middle" plane in the pbc case
    """

    L = 16
    R = 3

    N2 = 4
    E = 1.
    rc = E/4


    rng = np.random.RandomState(seed=5621)
    
    N = 10

    s = state.State()
    s.npart = N
    
    extent = [E/2 if bx else E for bx in direction]

    s.domain = domain.BaseDomainHalo(extent=extent)
    s.domain.boundary_condition = domain.BoundaryTypePeriodic()

    s.p = data.PositionDat()
    s.q = data.ParticleDat(ncomp=1)
    s.gid = data.ParticleDat(ncomp=1, dtype=ctypes.c_int64)
    
    if N == 1:
        s.p[0,:] = (halfmeps-0.5,0,0)
        s.q[0,0] = 1.0
    elif N == 2:
        s.p[0,:] = (halfmeps-0.5, 0.45 * E, 0)
        s.q[0,0] = 1.0
        s.p[1,:] = (halfmeps-0.5, -0.45 * E, 0)
        s.q[1,0] = -1.0       
    else:
        for px in range(N):
            for dimx in range(3):
                if direction[dimx]:
                    tpos = rng.uniform(low=-0.25, high=0.25)
                else:
                    tpos = rng.uniform(low=-0.5, high=0.5)

                s.p[px, dimx] = tpos

        s.q[:] = rng.uniform(low=-0.5, high=0.5, size=(N, 1))
        bias = np.sum(s.q[:N:]) / N
        s.q[:] -= bias
    

    print("\n")
    print(s.p[:N:,:])
    print("total charge", np.sum(s.q[:N:, 0]))
    print("----")

    s.gid[:, 0] = np.arange(0, N)

    mcs = kmc_dirichlet_boundary.MirrorChargeSystem(direction, s, 'p', 'q', 'gid')
    ms = mcs.mirror_state

    print(ms.p[:2*N:, :])
    print(ms.q[:2*N:, 0])

    ms.scatter_data_from(0)
    
    bcs = 'pbc'
 
    kmc_fmm = KMCFMM(positions=ms.p, charges=ms.q, 
        domain=ms.domain, r=R, l=L, boundary_condition=bcs)
    kmc_fmm.initialise()
    
    dipole_mag = np.zeros(3)
    for px in range(N*2):
        dipole_mag[:] += ms.p[px, :] * ms.q[px, 0]

    print("computed dipole magnitude", dipole_mag)

    
    if direction[0]: 
        plane_vector_1 = (0,1,0)
        plane_vector_2 = (0,0,1)
        plane_vector_3 = (1,0,0)
    elif direction[1]: 
        plane_vector_1 = (1,0,0)
        plane_vector_2 = (0,0,1)
        plane_vector_3 = (0,1,0)
    elif direction[2]: 
        plane_vector_1 = (1,0,0)
        plane_vector_2 = (0,1,0)
        plane_vector_3 = (0,0,1)
    else:
        raise RuntimeError('failed to set plane vectors')
    
    plane_vector_1 = np.array(plane_vector_1, dtype=REAL)
    plane_vector_2 = np.array(plane_vector_2, dtype=REAL)
    plane_vector_3 = np.array(plane_vector_3, dtype=REAL)

    X,Y = np.meshgrid(
        np.linspace(-halfmeps*E, halfmeps*E, N2),
        np.linspace(-halfmeps*E, halfmeps*E, N2),
    )
    X = X.ravel()
    Y = Y.ravel()

    eval_points = np.zeros((3*N2*N2, 3), dtype=REAL)
    
    for px in range(N2*N2):
        tmp = X[px] * plane_vector_1 + Y[px] * plane_vector_2
        eval_points[px, :] = tmp
        eval_points[px + N2*N2, :] = tmp + halfmeps * E * plane_vector_3
        eval_points[px + 2*N2*N2, :] = tmp - halfmeps * E * plane_vector_3


    

    kmc_field = kmc_fmm.eval_field(eval_points)
    
    for px in range(kmc_field.shape[0]):
        print(eval_points[px, :], kmc_field[px])


    

    
    N2 = 50
    

    X,Y = np.meshgrid(
        np.linspace(-halfmeps*E, halfmeps*E, N2),
        np.linspace(-halfmeps*E, halfmeps*E, N2),
    )
    X = X.ravel()
    Y = Y.ravel()

    eval_points = np.zeros((N2*N2, 3), dtype=REAL)
    
    for px in range(N2*N2):
        tmp = X[px] * plane_vector_3 + Y[px] * plane_vector_1
        eval_points[px, :] = tmp

    kmc_field = kmc_fmm.eval_field(eval_points)

    import pyevtk.hl
    pyevtk.hl.pointsToVTK('/tmp/foo', X, Y, kmc_field, None)

    
    print("done plane 0")

    X,Y = np.meshgrid(
        np.linspace(-halfmeps*E, halfmeps*E, N2),
        np.linspace(-halfmeps*E, halfmeps*E, N2),
    )
    X = X.ravel()
    Y = Y.ravel()

    eval_points = np.zeros((N2*N2, 3), dtype=REAL)
    
    for px in range(N2*N2):
        tmp = X[px] * plane_vector_2 + Y[px] * plane_vector_1
        eval_points[px, :] = tmp + halfmeps * E * plane_vector_3

    kmc_field = kmc_fmm.eval_field(eval_points)


    print("done plane 1")

    pyevtk.hl.pointsToVTK('/tmp/foo2', kmc_field, Y, X, None)








    return
    err = np.linalg.norm(kmc_field, np.inf)
    assert err < 3*10.**-5








