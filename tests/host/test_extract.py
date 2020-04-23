from __future__ import print_function, division

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"

import pytest, ctypes, math
import numpy as np
from itertools import product

np.set_printoptions(linewidth=200)

from ppmd import *
from ppmd.coulomb.fmm import *
from ppmd.coulomb.ewald_half import *
from scipy.special import sph_harm, lpmv
import time

from math import *

from mpi4py import MPI
MPISIZE = MPI.COMM_WORLD.Get_size()
MPIRANK = MPI.COMM_WORLD.Get_rank()
MPIBARRIER = MPI.COMM_WORLD.Barrier
DEBUG = True
SHARED_MEMORY = 'omp'

from coulomb_kmc import *

_PROFILE = common.PROFILE

REAL = ctypes.c_double
INT64 = ctypes.c_int64

ox_range = tuple(range(-1, 2))

from kmc_test_common import *


def _direct_chooser(bc, domain, L):
    if bc == 'free_space':
        return kmc_direct.FreeSpaceDirect()
    elif bc == '27':
        return kmc_direct.NearestDirect(float(domain.extent[0]))
    elif bc == 'pbc':
        return kmc_direct.PBCDirect(float(domain.extent[0]), domain, L)


@pytest.mark.skipif('MPISIZE > 1')
@pytest.mark.parametrize("BC", ('free_space', '27', 'pbc'))
def test_propose_extract_1(BC):
    """
    Tests proposed moves one by one against direct calculation.
    """


    L = 16

    N = 20
    E = 2*3.1416

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()
    A.npart = N

    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)
    A.GID = data.ParticleDat(ncomp=1, dtype=INT64)

    rng = np.random.RandomState(3251)

    pi = rng.uniform(low=-0.5*E, high=0.5*E, size=(N, 3))
    qi = np.zeros((N,1), REAL)
    assert N % 2 == 0
    for px in range(N):
        qi[px, 0] = (-1)**px
    
    gi = np.arange(N).reshape((N, 1))

    with A.modify() as m:
        if MPIRANK == 0:
            m.add({
                A.P: pi[:, :],
                A.Q: qi[:, :],
                A.GID: gi[:, :]
            })


    kmc = kmc_fmm.KMCFMM(
        A.P, A.Q, A.domain, r=3, l=12, max_move=1.0, boundary_condition=BC)
    kmc.initialise()


    direct = _direct_chooser(BC, A.domain, L)
    
    # check initial energy agrees
    phi_direct_0 = direct(N, pi, qi)
    err = abs(kmc.energy - phi_direct_0) / abs(phi_direct_0)
    assert err < 10.**-5
    

    
    for testx in range(20):

        # find a +ve/-ve pair of charges

        num_remove = rng.randint(1, 10)
        remove_inds = []
        available = set(range(N))
        
        for tx in range(num_remove):
            ind = rng.randint(0, N)
            while((A.Q[ind, 0] < 0) or (ind not in available)):
                ind = rng.randint(0, N)
        
            remove_inds.append(ind)
            available.remove(ind)

        for tx in range(num_remove):
            ind = rng.randint(0, N)
            while((A.Q[ind, 0] > 0) or (ind not in available)):

                ind = rng.randint(0, N)
            remove_inds.append(ind)
            available.remove(ind)

        assert len(remove_inds) == 2*num_remove

        gids = [int(A.GID[gx, 0]) for gx in remove_inds]

        diff_extractor = kmc.propose_extract(remove_inds)

        inds = set(range(N))
        for gx in gids:
            inds.remove(gx)
        inds = np.array(tuple(inds), 'int')


        # direct extract energy
        phi_direct_1 = direct(len(inds), pi[inds, :], qi[inds, :])
        diff_direct = phi_direct_1 - phi_direct_0
        

        err = abs(diff_extractor - diff_direct) / abs(diff_direct)

        assert err < 10.**-5 or abs(diff_extractor - diff_direct) < 10.**-5

        diff_extractor2 = kmc.propose_extract((remove_inds, remove_inds))
        
        assert abs(diff_extractor2[0] - diff_extractor) < 10.**-12
        assert abs(diff_extractor2[1] - diff_extractor) < 10.**-12

        assert err < 3.1*10.**-4


@pytest.mark.skipif('MPISIZE > 1')
@pytest.mark.parametrize("BC", ('free_space', '27', 'pbc'))
def test_extract_1(BC):
    """
    Tests extraction of charges, assumes propose_extract works (i.e passes above test).
    """


    L = 12

    N = 20
    E = 2*3.1416

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()
    A.npart = N

    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)
    A.GID = data.ParticleDat(ncomp=1, dtype=INT64)



    rng = np.random.RandomState(3251)

    pi = rng.uniform(low=-0.5*E, high=0.5*E, size=(N, 3))
    qi = np.zeros((N,1), REAL)
    assert N % 2 == 0
    for px in range(N):
        qi[px, 0] = (-1)**px
    
    gi = np.arange(N).reshape((N, 1))

    with A.modify() as m:
        if MPIRANK == 0:
            m.add({
                A.P: pi[:, :],
                A.Q: qi[:, :],
                A.GID: gi[:, :]
            })


    kmc = kmc_fmm.KMCFMM(
        A.P, A.Q, A.domain, r=3, l=12, max_move=1.0, boundary_condition=BC)
    kmc.initialise()


    direct = _direct_chooser(BC, A.domain, L)
    
    # check initial energy agrees
    phi_direct_0 = direct(N, pi, qi)
    err = abs(kmc.energy - phi_direct_0) / abs(phi_direct_0)
    assert err < 10.**-5
    

    
    for testx in range(70):
        # find a +ve/-ve pair of charges

        num_remove = rng.randint(1, 10)
        remove_inds = []
        available = set(range(N))
        
        for tx in range(num_remove):
            ind = rng.randint(0, N)
            while((A.Q[ind, 0] < 0) or (ind not in available)):
                ind = rng.randint(0, N)
        
            remove_inds.append(ind)
            available.remove(ind)

        for tx in range(num_remove):
            ind = rng.randint(0, N)
            while((A.Q[ind, 0] > 0) or (ind not in available)):

                ind = rng.randint(0, N)
            remove_inds.append(ind)
            available.remove(ind)

        assert len(remove_inds) == 2*num_remove


        gids = [int(A.GID[gx, 0]) for gx in remove_inds]

        diff_extractor = kmc.propose_extract(remove_inds)
        kmc.extract(remove_inds)

        inds = set(range(N))
        for gx in gids:
            inds.remove(gx)
        inds = np.array(tuple(inds), 'int')


        # direct extract energy
        phi_correct = phi_direct_0 + diff_extractor
        phi_test = kmc.energy
        err = abs(phi_test - phi_correct) / abs(phi_correct)


        #print(err, phi_correct, phi_test)

        assert err < 3*10.**-4

        with A.modify() as m:
            m.remove(tuple(range(A.npart_local)))
        assert A.npart == 0

        with A.modify() as m:
            if MPIRANK == 0:
                m.add({
                    A.P: pi[:, :],
                    A.Q: qi[:, :],
                    A.GID: gi[:, :]
                })

        kmc.initialise()
        err = abs(kmc.energy - phi_direct_0) / abs(phi_direct_0)
        assert err < 10.**-5


    
@pytest.mark.skipif('MPISIZE > 1')
@pytest.mark.parametrize("BC", ('free_space', '27', 'pbc'))
def test_extract_2(BC):
    """
    Tests extraction of charges, assumes propose_extract works (i.e passes above test).
    """


    L = 12

    N = 20
    E = 2*3.1416

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()
    A.npart = N

    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)
    A.GID = data.ParticleDat(ncomp=1, dtype=INT64)



    rng = np.random.RandomState(3251)

    pi = rng.uniform(low=-0.5*E, high=0.5*E, size=(N, 3))
    qi = np.zeros((N,1), REAL)
    assert N % 2 == 0
    for px in range(N):
        qi[px, 0] = (-1)**px
    
    gi = np.arange(N).reshape((N, 1))

    with A.modify() as m:
        if MPIRANK == 0:
            m.add({
                A.P: pi[:, :],
                A.Q: qi[:, :],
                A.GID: gi[:, :]
            })


    kmc = kmc_fmm.KMCFMM(
        A.P, A.Q, A.domain, r=3, l=12, max_move=1.0, boundary_condition=BC)
    kmc.initialise()


    direct = _direct_chooser(BC, A.domain, L)
    
    # check initial energy agrees
    phi_direct_0 = direct(N, pi, qi)
    err = abs(kmc.energy - phi_direct_0) / abs(phi_direct_0)
    assert err < 10.**-5
    

    
    for testx in range(10):
        # find a +ve/-ve pair of charges
        
        N = A.npart_local
        remove_inds = []
        available = set(range(N))
        
        ind = rng.randint(0, N)
        while((A.Q[ind, 0] < 0) or (ind not in available)):
            ind = rng.randint(0, N)
    
        remove_inds.append(ind)
        available.remove(ind)

        ind = rng.randint(0, N)
        while((A.Q[ind, 0] > 0) or (ind not in available)):

            ind = rng.randint(0, N)
        remove_inds.append(ind)
        available.remove(ind)

        assert len(remove_inds) == 2


        gids = [int(A.GID[gx, 0]) for gx in remove_inds]
        
        kmc_phi_0 = kmc.energy
        diff_extractor = kmc.propose_extract(remove_inds)
        kmc.extract(remove_inds)
        kmc_phi_1 = kmc.energy

        assert abs(kmc_phi_1 - kmc_phi_0 - diff_extractor) < 10.**-8

        correct = direct(A.npart_local, A.P.view.copy(), A.Q.view.copy())

        assert abs(correct - kmc_phi_1) < 10.**-5





class _FMMSolver:
    def __init__(self, extent, L, R):
        self.A = state.State()
        self.A.domain = domain.BaseDomainHalo(extent=extent)
        self.A.domain.boundary_condition = domain.BoundaryTypePeriodic()

        self.A.P = data.PositionDat(ncomp=3)
        self.A.Q = data.ParticleDat(ncomp=1)
        self.A.U = data.ParticleDat(ncomp=1)
        self.A.GID = data.ParticleDat(ncomp=1, dtype=INT64)

        self.FMM = PyFMM(self.A.domain, r=R, l=L, free_space=False)
    

    def __call__(self, S):

        with self.A.modify() as m:
            m.add({
                self.A.P   : S.P.view.copy(),
                self.A.Q   : S.Q.view.copy(),
                self.A.GID : S.GID.view.copy(),
            })
        
        u = self.FMM(self.A.P, self.A.Q, None, self.A.U)
        up = self.A.U.view.copy()
        ug = self.A.GID.view.copy()

        with self.A.modify() as m:
            m.remove(tuple(range(self.A.npart_local)))

        return u, up, ug


@pytest.mark.skipif('MPISIZE > 1')
def test_extract_3():
    """
    Tests extraction of charges, assumes propose_extract works (i.e passes above test).
    """


    L = 16
    R = 3
    
    HN = 10
    N = 2 * HN
    E = 2*3.1416

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)
    A.GID = data.ParticleDat(ncomp=1, dtype=INT64)



    rng = np.random.RandomState(3251)
    
    pih = np.zeros((HN, 3), REAL)
    pih[:, 0] = rng.uniform(low=-0.5*E, high=0.5*E, size=(HN))
    pih[:, 1] = rng.uniform(low=-0.5*E, high=0.5*E, size=(HN))
    pih[:, 2] = rng.uniform(low=-0.5*E, high=0.0,   size=(HN))
    
    pi = np.zeros((N, 3), REAL)
    qi = np.zeros((N, 1), REAL)
    gi = np.zeros((N, 1), INT64)

    assert N % 2 == 0
    for px in range(HN):
        qi[px*2, 0] = 1.0
        qi[px*2+1, 0] = -1.0
        pi[px*2, :] = pih[px, :].copy()
        pi[px*2+1, :] = pih[px, :].copy()
        pi[px*2+1, 2] *= -1.0
        gi[px*2, 0] = px*2
        gi[px*2+1, 0] = px*2+1
    

    with A.modify() as m:
        if MPIRANK == 0:
            m.add({
                A.P: pi[:, :],
                A.Q: qi[:, :],
                A.GID: gi[:, :]
            })
    

    kmc = kmc_fmm.KMCFMM(
        A.P, A.Q, A.domain, r=R, l=L, max_move=1.0, boundary_condition='pbc', mirror_direction=(False, False, True))
    kmc.initialise()


    direct = _direct_chooser('pbc', A.domain, L)
    
    fmm_solver = _FMMSolver((E, E, E), L, R)

    
    # check initial energy agrees
    phi_direct_0 = direct(N, pi, qi)
    err = abs(kmc.energy - phi_direct_0) / abs(phi_direct_0)
    assert err < 10.**-5
    
    
    for testx in range(HN):
        # find a +ve/-ve pair of charges

        n = A.npart_local

        ut = kmc.get_energy(np.atleast_2d(np.arange(n)).reshape((n, 1)))

        u, up, ug = fmm_solver(A)

        for ix in range(A.npart_local):
            assert ug[ix] == A.GID[ix]

            err = abs(ut[ix] - up[ix])
            err = err if abs(up[ix]) == 0 else err / abs(up[ix])

            assert err < 10.**-5


        px = rng.randint(A.npart_local)
        gid_i = int(A.GID[px, 0])
        gid_j = gid_i + 1 if gid_i % 2 == 0 else gid_i - 1
        py = int(np.where(A.GID.view[:, 0] == gid_j)[0])
        

        assert abs(A.P[px, 0] - A.P[py, 0]) < 10.**-14
        assert abs(A.P[px, 1] - A.P[py, 1]) < 10.**-14
        assert abs(A.P[px, 2] + A.P[py, 2]) < 10.**-14

        kmc.extract((px, py))






@pytest.mark.parametrize("BC", ('free_space', '27', 'pbc'))
def test_get_energy_1(BC):
    """
    Test the get_energy interfaces
    """


    L = 12

    N = 200
    E = 2*3.1416

    M = 10

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()
    A.npart = N

    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)
    A.U = data.ParticleDat(ncomp=1)
    A.E = data.ParticleDat(ncomp=1)
    A.M = data.ParticleDat(ncomp=1, dtype=INT64)
    A.GID = data.ParticleDat(ncomp=1, dtype=INT64)

    rng = np.random.RandomState(3251 * (MPIRANK+1))

    pi = rng.uniform(low=-0.5*E, high=0.5*E, size=(N, 3))
    qi = np.zeros((N,1), REAL)
    for px in range(N):
        qi[px, 0] = (-1)**px

    bias = np.sum(qi)
    qi -= bias/N
    
    gi = np.arange(N).reshape((N, 1))

    with A.modify() as m:
        if MPIRANK == 0:
            m.add({
                A.P: pi[:, :],
                A.Q: qi[:, :],
                A.GID: gi[:, :]
            })

    kmc = kmc_fmm.KMCFMM(
        A.P, A.Q, A.domain, r=3, l=12, max_move=1.0, boundary_condition=BC)
    
    
    fmm_bc = {
        'free_space': True,
        '27': '27',
        'pbc': False
    }[BC]

    FMM = PyFMM(A.domain, r=kmc.fmm.R, l=L, free_space=fmm_bc)


    for testx in range(20):

        
        pi = rng.uniform(low=-0.5*E, high=0.5*E, size=(A.npart_local, 3))
    
        with A.P.modify_view() as mv:
            mv[:] = pi.copy()

        kmc.initialise()

        FMM(A.P, A.Q, potential=A.U)
        
        Ma = min(A.npart_local, M)
        ids = rng.permutation(A.npart_local)[:Ma].reshape((Ma, 1))


        to_test_1 = kmc.get_energy(ids)

        for ix, px in enumerate(ids):
            assert abs(to_test_1[ix] - A.U[px, 0]) < 10.**-6
        
        with A.M.modify_view() as mv:
            mv[:] = 0
            mv[ids, 0] = 1
        
        kmc.get_energy_with_dats(A.M, A.E)

        assert np.linalg.norm(A.E[ids, 0].ravel() - to_test_1.ravel(), np.inf) < 10.**-14

        




@pytest.mark.skipif('MPISIZE > 1')
@pytest.mark.parametrize("BC", ('pbc', ))
def test_propose_extract_split_1(BC):
    """
    Tests proposed moves one by one against direct calculation.
    """


    L = 12

    N = 20
    E = 2*3.1416


    NEAR = state.State(
        domain=domain.BaseDomainHalo(
            extent=(E,E,E),
            boundary_condition=domain.BoundaryTypePeriodic()
        ),
        particle_dats={
            'P': data.PositionDat(ncomp=3),
            'Q': data.ParticleDat(ncomp=1),
            'GID': data.ParticleDat(ncomp=1, dtype=INT64),
        }
    )

    FAR = state.State(
        domain=domain.BaseDomainHalo(
            extent=(E,E,E),
            boundary_condition=domain.BoundaryTypePeriodic()
        ),
        particle_dats={
            'P': data.PositionDat(ncomp=3),
            'Q': data.ParticleDat(ncomp=1),
            'GID': data.ParticleDat(ncomp=1, dtype=INT64),
        }
    )

    rng = np.random.RandomState(3251)

    pi = rng.uniform(low=-0.5*E, high=0.5*E, size=(N, 3))
    qi = np.zeros((N,1), REAL)
    assert N % 2 == 0
    for px in range(N):
        qi[px, 0] = (-1)**px
    
    gi = np.arange(N).reshape((N, 1))


    for sx in (FAR, NEAR):
        with sx.modify() as m:
            if MPIRANK == 0:
                m.add({
                    sx.P: pi[:, :],
                    sx.Q: qi[:, :],
                    sx.GID: gi[:, :]
                })

    FAR_KMC = kmc_fmm.KMCFMM(FAR.P, FAR.Q, FAR.domain, r=3, l=L, max_move=1.0, boundary_condition='ff-only')
    NEAR_KMC = kmc_fmm.KMCFMM(NEAR.P, NEAR.Q, NEAR.domain, r=3, l=L, max_move=1.0, boundary_condition='27')


    FAR_KMC.initialise()
    NEAR_KMC.initialise()

    direct = _direct_chooser(BC, NEAR.domain, L)
    phi_direct_0 = direct(N, pi, qi)

    
    for testx in range(20):

        # find a +ve/-ve pair of charges

        num_remove = rng.randint(1, 10)
        remove_inds = []
        available = set(range(N))
        
        for tx in range(num_remove):
            ind = rng.randint(0, N)
            while((NEAR.Q[ind, 0] < 0) or (ind not in available)):
                ind = rng.randint(0, N)
        
            remove_inds.append(ind)
            available.remove(ind)

        for tx in range(num_remove):
            ind = rng.randint(0, N)
            while((NEAR.Q[ind, 0] > 0) or (ind not in available)):

                ind = rng.randint(0, N)
            remove_inds.append(ind)
            available.remove(ind)

        assert len(remove_inds) == 2*num_remove

        gids = [int(NEAR.GID[gx, 0]) for gx in remove_inds]

        NEAR_diff = NEAR_KMC.propose_extract(remove_inds)
        FAR_diff = FAR_KMC.propose_extract(remove_inds)

        inds = set(range(N))
        for gx in gids:
            inds.remove(gx)
        inds = np.array(tuple(inds), 'int')


        # direct extract energy
        phi_direct_1 = direct(len(inds), pi[inds, :], qi[inds, :])
        diff_direct = phi_direct_1 - phi_direct_0


        to_test = NEAR_diff + FAR_diff
        err_abs = abs(to_test - diff_direct)
        err = err_abs / abs(diff_direct)
        assert err < 10.**-5 or err_abs < 10**-5


        NEAR_diff2 = NEAR_KMC.propose_extract((remove_inds, remove_inds))
        FAR_diff2 = FAR_KMC.propose_extract((remove_inds, remove_inds))
        
        to_test2 = NEAR_diff2 + FAR_diff2
        assert abs(to_test2[0] - to_test) < 10.**-12
        assert abs(to_test2[1] - to_test) < 10.**-12



    
@pytest.mark.skipif('MPISIZE > 1')
@pytest.mark.parametrize("BC", ('pbc', ))
def test_extract_split_1(BC):
    """
    Tests extraction of charges, assumes propose_extract works (i.e passes above test).
    """


    L = 16
    N = 20
    E = 2*3.1416


    NEAR = state.State(
        domain=domain.BaseDomainHalo(
            extent=(E,E,E),
            boundary_condition=domain.BoundaryTypePeriodic()
        ),
        particle_dats={
            'P': data.PositionDat(ncomp=3),
            'Q': data.ParticleDat(ncomp=1),
            'GID': data.ParticleDat(ncomp=1, dtype=INT64),
        }
    )

    FAR = state.State(
        domain=domain.BaseDomainHalo(
            extent=(E,E,E),
            boundary_condition=domain.BoundaryTypePeriodic()
        ),
        particle_dats={
            'P': data.PositionDat(ncomp=3),
            'Q': data.ParticleDat(ncomp=1),
            'GID': data.ParticleDat(ncomp=1, dtype=INT64),
        }
    )



    rng = np.random.RandomState(3251)

    pi = rng.uniform(low=-0.5*E, high=0.5*E, size=(N, 3))
    qi = np.zeros((N,1), REAL)
    assert N % 2 == 0
    for px in range(N):
        qi[px, 0] = (-1)**px
    
    gi = np.arange(N).reshape((N, 1))


    for sx in (NEAR, FAR):
        with sx.modify() as m:
            if MPIRANK == 0:
                m.add({
                    sx.P: pi[:, :],
                    sx.Q: qi[:, :],
                    sx.GID: gi[:, :]
                })

    FAR_KMC = kmc_fmm.KMCFMM(FAR.P, FAR.Q, FAR.domain, r=3, l=L, max_move=1.0, boundary_condition='ff-only')
    NEAR_KMC = kmc_fmm.KMCFMM(NEAR.P, NEAR.Q, NEAR.domain, r=3, l=L, max_move=1.0, boundary_condition='27')

    FAR_KMC.initialise()
    NEAR_KMC.initialise()

    direct = _direct_chooser(BC, FAR.domain, L)
    phi_direct_0 = direct(N, pi, qi)
    
    for testx in range(10):
        # find a +ve/-ve pair of charges
        
        N = FAR.npart_local
        remove_inds = []
        available = set(range(N))
        
        ind = rng.randint(0, N)
        while((FAR.Q[ind, 0] < 0) or (ind not in available)):
            ind = rng.randint(0, N)
    
        remove_inds.append(ind)
        available.remove(ind)

        ind = rng.randint(0, N)
        while((FAR.Q[ind, 0] > 0) or (ind not in available)):

            ind = rng.randint(0, N)
        remove_inds.append(ind)
        available.remove(ind)

        assert len(remove_inds) == 2


        gids = [int(FAR.GID[gx, 0]) for gx in remove_inds]
        

        kmc_phi_0 = FAR_KMC.energy + NEAR_KMC.energy
        FAR_diff = FAR_KMC.propose_extract(remove_inds)
        NEAR_diff = NEAR_KMC.propose_extract(remove_inds)
        diff_extractor = FAR_diff + NEAR_diff

        FAR_KMC.extract(remove_inds)
        NEAR_KMC.extract(remove_inds)

        kmc_phi_1 = FAR_KMC.energy + NEAR_KMC.energy

        assert abs(kmc_phi_1 - kmc_phi_0 - diff_extractor) < 10.**-8

        correct = direct(NEAR.npart_local, NEAR.P.view.copy(), NEAR.Q.view.copy())

        assert abs(correct - kmc_phi_1) < 10.**-5











