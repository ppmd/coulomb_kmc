


import pytest

from ppmd import *
from ppmd.coulomb.fmm import *
from ppmd.coulomb.ewald_half import *

import ctypes

REAL = ctypes.c_double
INT64 = ctypes.c_int64

import numpy as np


from kmc_test_common import *

from mpi4py import MPI
MPISIZE = MPI.COMM_WORLD.Get_size()
MPIRANK = MPI.COMM_WORLD.Get_rank()

def test_free_space_1():

    FSD = FreeSpaceDirect()

    for testx in range(500):

        rng = np.random.RandomState(seed=(MPIRANK+1)*93573)
        N = rng.randint(1, 100)
        
        ppi = np.zeros((N, 3), REAL)
        qi = np.zeros((N, 1), REAL)

        def _direct():
            _phi_direct = 0.0
            # compute phi from image and surrounding 26 cells
            for ix in range(N):
                for jx in range(ix+1, N):
                    rij = np.linalg.norm(ppi[jx,:] - ppi[ix,:])
                    _phi_direct += qi[ix, 0] * qi[jx, 0] / rij
            return _phi_direct


        ppi[:] = rng.uniform(-1.0, 1.0, (N,3))
        qi[:] = rng.uniform(-1.0, 1.0, (N,1))

        phi_py = _direct()
        phi_c = FSD(N, ppi, qi)

        rel = abs(phi_py)
        rel = 1.0 if rel == 0 else rel
        err = abs(phi_py - phi_c) / rel
        assert err < 10.**-14



ox_range = tuple(range(-1, 2))

def test_nearest_1():

    E = 39.
    ND = NearestDirect(E)


    for testx in range(max(10, 20//MPISIZE)):

        rng = np.random.RandomState(seed=(MPIRANK+1)*93573)
        N = rng.randint(1, 100)


        ppi = np.zeros((N, 3), REAL)
        qi = np.zeros((N, 1), REAL)

        def _direct():
            _phi_direct = 0.0
            # compute phi from image and surrounding 26 cells
            for ix in range(N):
                for jx in range(ix+1, N):
                    rij = np.linalg.norm(ppi[jx,:] - ppi[ix,:])
                    _phi_direct += qi[ix, 0] * qi[jx, 0] / rij

                for jx in range(N):
                    for ox in product(ox_range, ox_range, ox_range):
                        if ox[0] != 0 or ox[1] != 0 or ox[2] != 0:
                            rij = np.linalg.norm(ppi[jx,:] - ppi[ix,:] + (E*np.array(ox)))
                            _phi_direct += 0.5 * qi[ix, 0] * qi[jx, 0] / rij

            return _phi_direct


        ppi[:] = rng.uniform(-1.0, 1.0, (N,3))
        qi[:] = rng.uniform(-1.0, 1.0, (N,1))

        phi_py = _direct()
        phi_c = ND(N, ppi, qi)

        rel = abs(phi_py)
        rel = 1.0 if rel == 0 else rel
        err = abs(phi_py - phi_c) / rel

        assert err < 10.**-13


def test_pbc_1():

    E = 19.
    L = 16
    N = 10
    rc = E/4

    rng  = np.random.RandomState(seed=8123)

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()
    A.npart = N
    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)
    A.F = data.ParticleDat(ncomp=3)
    A.G = data.ParticleDat(ncomp=1, dtype=INT64)

    pi = np.zeros((N, 3), REAL)
    qi = np.zeros((N, 1), REAL)
    gi = np.zeros((N, 1), INT64)
    
    gi[:, 0] = np.arange(N)
    pi[:] = rng.uniform(-0.5*E, 0.5*E, (N, 3))
    for px in range(N):
        qi[px,0] = (-1.0)**(px+1)
    bias = np.sum(qi)/N
    qi[:] -= bias

    A.P[:] = pi
    A.Q[:] = qi
    A.G[:] = gi

    A.scatter_data_from(0)

    EWALD = EwaldOrthoganalHalf(domain=A.domain, real_cutoff=rc, shared_memory='omp', eps=10.**-8)
    FMM = PyFMM(A.domain, N=N, free_space=False, r=3, l=L)
    PBCD = PBCDirect(E, A.domain, L)

    def _check1():

        phi_e = EWALD(positions=A.P, charges=A.Q, forces=A.F)
        phi_f = FMM(positions=A.P, charges=A.Q)

        rel = abs(phi_f)
        rel = 1.0 if rel == 0 else rel
        err = abs(phi_e - phi_f) / rel
        assert err < 10.**-4
        return phi_f
    
    phi_f = _check1()

    phi_c = PBCD(N, pi, qi)
    rel = abs(phi_f)
    rel = 1.0 if rel == 0 else rel
    err = abs(phi_c - phi_f) / rel    
    assert err < 10.**-5
 
    for testx in range(100):

        pi[:] = rng.uniform(-0.5*E, 0.5*E, (N, 3))
        with A.P.modify_view() as m:
            for px in range(A.npart_local):
                g = A.G[px, 0]
                A.P[px, :] = pi[g, :]
    
        phi_f = _check1()

        phi_c = PBCD(N, pi, qi)
        rel = abs(phi_f)
        rel = 1.0 if rel == 0 else rel
        err = abs(phi_c - phi_f) / rel    
        assert err < 10.**-4
         

    FMM.free()












@pytest.mark.parametrize("BC", (BCType.FREE_SPACE, BCType.NEAREST, BCType.PBC))
def test_pair_direct_dats_1(BC):
    
    E = 19.
    L = 16
    N = 100
    M = 8
    rc = E/4

    rng  = np.random.RandomState(seed=8123)

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)
    A.GF = data.ParticleDat(ncomp=1, dtype=INT64)
    A.GC = data.ParticleDat(ncomp=1, dtype=INT64)
    A.GP = data.ParticleDat(ncomp=M*3, dtype=REAL)
    A.GQ = data.ParticleDat(ncomp=M, dtype=REAL)
    A.GU = data.ParticleDat(ncomp=M, dtype=REAL)
    
    
    Pi = rng.uniform(low=-0.5*E, high=0.5*E, size=(N, 3))
    Qi = rng.uniform(low=-1.0, high=1.0, size=(N, 1))
    bias = np.sum(Qi) / N
    Qi -= bias
    GFi = np.ones((N, 1))
    GCi = rng.randint(1, M+1, (N, 1))
    GPi = rng.uniform(low=-0.5*E, high=0.5*E, size=(N, M*3))
    GQi = rng.uniform(low=-1.0, high=1.0, size=(N, M))

    with A.modify() as m:
        if MPIRANK == 0:
            m.add({
                A.P: Pi,
                A.Q: Qi,
                A.GF: GFi,
                A.GC: GCi,
                A.GP: GPi,
                A.GQ: GQi
            })

    
    PDFD = PairDirectFromDats(A.domain, BC, L, M)
    PDFD(
        A.GF,
        A.P,
        A.Q,
        A.GC,
        A.GP,
        A.GQ,
        A.GU
    )

    if BC == BCType.FREE_SPACE:
        DIRECT = FreeSpaceDirect()
        TOL = 10.**-15
    elif BC == BCType.NEAREST:
        DIRECT = NearestDirect(E)   
        TOL = 10.**-15
    elif BC == BCType.PBC:
        DIRECT = PBCDirect(E, A.domain, L)
        TOL = 10.**-5

    for px in range(A.npart_local):
        for gx in range(A.GC[px, 0]):

            tmp_positions = np.zeros((2,3), REAL)
            tmp_charges = np.zeros((2,1), REAL)

            tmp_positions[0, :] = A.P[px, :]
            tmp_positions[1, :] = A.GP[px, gx*3:(gx+1)*3:]

            tmp_charges[0, 0] = A.Q[px, 0]
            tmp_charges[1, 0] = A.GQ[px, gx]

            correct = DIRECT(2, tmp_positions, tmp_charges)
            to_test = A.GU[px, gx]
            err = abs(correct - to_test) / abs(correct)
            #print(err, correct, to_test)
            #import ipdb; ipdb.set_trace()
            assert err < TOL



    


def test_pair_direct_dats_2():
    # mirror mode test
    
    E = 19.
    L = 16
    N = 100
    M = 8
    rc = E/4

    BC = BCType.PBC

    rng  = np.random.RandomState(seed=8123)

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)
    A.GF = data.ParticleDat(ncomp=1, dtype=INT64)
    A.GC = data.ParticleDat(ncomp=1, dtype=INT64)
    A.GP = data.ParticleDat(ncomp=M*3, dtype=REAL)
    A.GQ = data.ParticleDat(ncomp=M, dtype=REAL)
    A.GU = data.ParticleDat(ncomp=M, dtype=REAL)
    
    Pi = np.zeros((N, 3), REAL)
    Pi[:, 0] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N))
    Pi[:, 1] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N))
    Pi[:, 2] = rng.uniform(low=-0.25*E, high=0.25*E, size=(N))

    Qi = rng.uniform(low=-1.0, high=1.0, size=(N, 1))
    bias = np.sum(Qi) / N
    Qi -= bias
    GFi = np.ones((N, 1))
    GCi = rng.randint(1, M+1, (N, 1))

    GPi = np.zeros((N, 3*M), REAL)
    GPi[:, 0::3] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N, M))
    GPi[:, 1::3] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N, M))
    GPi[:, 2::3] = rng.uniform(low=-0.25*E, high=0.25*E, size=(N, M))

    GQi = rng.uniform(low=-1.0, high=1.0, size=(N, M))

    with A.modify() as m:
        if MPIRANK == 0:
            m.add({
                A.P: Pi,
                A.Q: Qi,
                A.GF: GFi,
                A.GC: GCi,
                A.GP: GPi,
                A.GQ: GQi
            })

    
    PDFD = PairDirectFromDats(A.domain, BC, L, M, mirror_mode=True)
    PDFD(
        A.GF,
        A.P,
        A.Q,
        A.GC,
        A.GP,
        A.GQ,
        A.GU
    )

    DIRECT = PBCDirect(E, A.domain, L)
    TOL = 10.**-5

    for px in range(A.npart_local):
        for gx in range(A.GC[px, 0]):

            tmp_positions = np.zeros((4,3), REAL)
            tmp_charges = np.zeros((4,1), REAL)


            tmp_positions[0, :] = A.P[px, :]
            tmp_positions[1, :] = A.GP[px, gx*3:(gx+1)*3:]
            tmp_positions[2, :] = A.P[px, :]
            tmp_positions[3, :] = A.GP[px, gx*3:(gx+1)*3:]

            tmp_positions[:, 2] -= 0.25 * E
            tmp_positions[2, 2] *= -1.0
            tmp_positions[3, 2] *= -1.0

            tmp_charges[0, 0] = A.Q[px, 0]
            tmp_charges[1, 0] = A.GQ[px, gx]
            tmp_charges[2, 0] = -1.0 * A.Q[px, 0]
            tmp_charges[3, 0] = -1.0 * A.GQ[px, gx]


            correct = DIRECT(4, tmp_positions, tmp_charges)
            to_test = A.GU[px, gx]
            err = abs(correct - to_test) / abs(correct)
            #print(err, correct, to_test)
            assert err < TOL







