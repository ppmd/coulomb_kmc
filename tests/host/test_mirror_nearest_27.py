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
INT64 = ctypes.c_int64


@pytest.mark.skipif('True')
def test_kmc_fmm_nearest_27_1():
    """
    Tests proposed moves one by one against direct calculation.
    Considers the primary image and the nearest 27 neighbours.
    """
    direction = (True, False, False)

    eps = 10.**-5
    L = 12
    R = 3

    N = 2
    N2 = 2 * N
    E = 4.
    rc = E/4


    rng = np.random.RandomState(seed=1234)

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()
    A.npart = N2

    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)
    A.GID = data.ParticleDat(ncomp=1, dtype=INT64)
    A.MIRROR_MAP = data.ParticleDat(ncomp=2, dtype=INT64)


    B = state.State()
    B.domain = domain.BaseDomainHalo(extent=(E,E,E))
    B.domain.boundary_condition = domain.BoundaryTypePeriodic()
    B.npart = N2

    B.P = data.PositionDat(ncomp=3)
    B.Q = data.ParticleDat(ncomp=1)
    B.GID = data.ParticleDat(ncomp=1, dtype=INT64)
    B.MIRROR_MAP = data.ParticleDat(ncomp=2, dtype=INT64)


    half_extent = [E/2 if bx else E for bx in direction]

    S = state.State()
    S.npart = N
    S.domain = domain.BaseDomainHalo(extent=half_extent)
    S.domain.boundary_condition = domain.BoundaryTypePeriodic()
    S.P = data.PositionDat()
    S.Q = data.ParticleDat(ncomp=1)
    S.GID = data.ParticleDat(ncomp=1, dtype=ctypes.c_int64)

    for dimx in range(3):
        he = 0.25*E if direction[dimx] else 0.5*E
        S.P[:, dimx] = rng.uniform(low=-he, high=he, size=N)

    q_init = -1.0
    S.Q[0,0] = q_init * 1
    S.Q[1,0] = q_init * 1

    S.GID[:N, 0] = np.arange(N)

    MCS = kmc_dirichlet_boundary.MirrorChargeSystem(direction, S, 'P', 'Q', 'GID')
    MS = MCS.mirror_state
    
    A.P[:N2, :] = MS.P[:N2, :]
    A.Q[:N2, :] = MS.Q[:N2, :]
    A.GID[:N2, :] = MS.GID[:N2, :]
    A.MIRROR_MAP[:N2, :]  = MS.mirror_map[:N2, :]
    
    B.P[:N2, :] = MS.P[:N2, :]
    B.Q[:N2, :] = MS.Q[:N2, :]
    B.GID[:N2, :] = MS.GID[:N2, :]
    B.MIRROR_MAP[:N2, :]  = MS.mirror_map[:N2, :]

    A.scatter_data_from(0)
    B.scatter_data_from(0)

    bc = 'free_space'
    bc = '27'
    if bc == 'free_space':
        fmm_bc = True
        kmc_bc = 'free_space'
        iterset = (0,)
    else:
        fmm_bc = '27'
        kmc_bc = '27'
        iterset = (-1,0,1)
    
    print("B" * 60)
    print(B.P[:N2,:])
    print(B.Q[:N2,:])
    print("A" * 60)
    print(A.P[:N2,:])
    print(A.Q[:N2,:])


    kmc_fmm = KMCFMM(positions=A.P, charges=A.Q, domain=A.domain, r=R, l=L, 
        boundary_condition=kmc_bc, mirror_direction=direction)
    kmc_fmm.initialise()
    fmm = PyFMM(B.domain, N=N2, free_space=fmm_bc, r=kmc_fmm.fmm.R, l=kmc_fmm.fmm.L)

    def _direct():
        _phi_direct = fmm(positions=B.P, charges=B.Q)
        return _phi_direct
    
    
    def _direct2():
        tmp_energy = 0.0

        for ix in range(N2):
            peng = 0.0
            for ofx in product(iterset, iterset, iterset):
                offset = np.array([ex*ox for ex, ox in zip(B.domain.extent, ofx)])
                for jx in range(N2):
                    if not (ofx[0]==0 and ofx[1]==0 and ofx[2]==0 and ix==jx):
                        r = np.linalg.norm(B.P[ix,:] - (B.P[jx,:] + offset))
                        peng += B.Q[ix,0] * B.Q[jx,0] / r
            
            print("phi_0 ix= contrib", ix, peng)

            tmp_energy += peng
        return tmp_energy * 0.5


    def _a_a2():
        peng = 0.0
        for ix in range(N2):
            for ofx in product(iterset, iterset, iterset):
                offset = np.array([ex*ox for ex, ox in zip(B.domain.extent, ofx)])
                for jx in range(N2):
                    if ix in (0,2) and jx in (0,2):
                        continue
                        
                    elif not (ofx[0]==0 and ofx[1]==0 and ofx[2]==0 and ix==jx):
                        r = np.linalg.norm(B.P[ix,:] - (B.P[jx,:] + offset))
                        peng += B.Q[ix,0] * B.Q[jx,0] / r

        return peng * 0.5


    def _a_a_diff():
        peng = 0.0

        for ix in (0,2):
            for ofx in product(iterset, iterset, iterset):
                offset = np.array([ex*ox for ex, ox in zip(B.domain.extent, ofx)])
                for jx in (0,2):
                    if not (ofx[0]==0 and ofx[1]==0 and ofx[2]==0 and ix==jx):
                        r = np.linalg.norm(B.P[ix,:] - (B.P[jx,:] + offset))
                        peng += B.Q[ix,0] * B.Q[jx,0] / r
        return peng * 0.5    


    def _a_a():
        peng = 0.0
        for ix in (1,3):
            for ofx in product(iterset, iterset, iterset):
                offset = np.array([ex*ox for ex, ox in zip(B.domain.extent, ofx)])
                for jx in (1,3):
                    if not (ofx[0]==0 and ofx[1]==0 and ofx[2]==0 and ix==jx):
                        r = np.linalg.norm(B.P[ix,:] - (B.P[jx,:] + offset))
                        peng += B.Q[ix,0] * B.Q[jx,0] / r
        return peng * 0.5


    def _direct3(new_pos):
        tmp_energy = 0.0

        peng = 0.0
        for ofx in product(iterset, iterset, iterset):
            offset = np.array([ex*ox for ex, ox in zip(B.domain.extent, ofx)])
            for jx in range(N2):
                r = np.linalg.norm(new_pos[0] - (B.P[jx,:] + offset))
                peng += B.Q[0, 0] * B.Q[jx, 0] / r
        
        tmp_energy += peng

        return tmp_energy



    phi_direct = _direct2()
    phi_0 = phi_direct
    
    #err = (kmc_fmm.energy - phi_direct)
    #print("initial error", err, "kmc energy", kmc_fmm.energy, "fmm call", _direct(), "direct", phi_0)
    #assert err < 10.**-5


    def _mirror_pos(rpos):
        f = [-1.0 if dx else 1.0 for dx in direction]
        return (rpos[0] * f[0], rpos[1] * f[1], rpos[2] * f[2])

    def _make_prop_pos():
        p = [0,0,0]
        for dimx in range(3):
            he = 0 if direction[dimx] else 0.5*E
            p[dimx] = rng.uniform(low=-0.5*E, high=he, size=1)[0]
        print(p)
        return p

    
    def _b_bp(new_pos, old_pos):
        tmp_energy = 0.0

        charge_with_mirror = -1.0 * B.Q[0,0] * B.Q[0,0]
        charge_with_same   =        B.Q[0,0] * B.Q[0,0]

        for ofx in product(iterset, iterset, iterset):
            offset = np.array([ex*ox for ex, ox in zip(B.domain.extent, ofx)])

            r = np.linalg.norm(new_pos[0] - (old_pos[1] + offset))
            pe = charge_with_mirror / r
            tmp_energy += pe

            r = np.linalg.norm(new_pos[0] - (old_pos[0] + offset))
            pe = charge_with_same / r
            tmp_energy += pe

        return tmp_energy

    def _b_b():
        peng = 0.0
        for ix in (0,):
            for ofx in product(iterset, iterset, iterset):
                offset = np.array([ex*ox for ex, ox in zip(B.domain.extent, ofx)])
                for jx in (0, 2):
                    if not (ofx[0]==0 and ofx[1]==0 and ofx[2]==0 and ix==jx):
                        r = np.linalg.norm(B.P[ix,:] - (B.P[jx,:] + offset))
                        peng += B.Q[ix,0] * B.Q[jx,0] / r
        return peng


    def _bp_bp(pos):

        charge_with_mirror = -1.0 * B.Q[0,0] * B.Q[0,0]
        charge_with_same   =        B.Q[0,0] * B.Q[0,0]

        te = 0.0
        for ofx in product(iterset, iterset, iterset):
            offset = np.array([ex*ox for ex, ox in zip(B.domain.extent, ofx)])
            
            if not (ofx[0] == 0 and ofx[1] == 0 and ofx[2] == 0):
                # interaction with the new mirrors
                r = np.linalg.norm(pos[0] - (pos[1] + offset))
                te += charge_with_mirror / r
                
                # interaction with the new non-mirrors
                r = np.linalg.norm(pos[0] - (pos[0] + offset))
                te += charge_with_same / r

        r = np.linalg.norm(pos[0] - pos[1])
        te += charge_with_mirror / r
        return te


    def _u0(ix):
        tmp_energy = 0.0
        px = B.P[ix, :]

        peng = 0.0
        for ofx in product(iterset, iterset, iterset):
            offset = np.array([ex*ox for ex, ox in zip(B.domain.extent, ofx)])
            for jx in range(N2):

                if not (jx == ix and ofx[0] == 0 and ofx[1] == 0 and ofx[2] == 0):
                    r = np.linalg.norm(px - (B.P[jx,:] + offset))
                    peng += B.Q[ix, 0] * B.Q[jx, 0] / r
                else:
                    print("SKIPPED", ix, jx, ofx)
        tmp_energy += peng

        return tmp_energy 


    pid = 0
    pos = np.array((-1.2, 0.0, 0.0))
    pos = _make_prop_pos()
    mpos = np.array(_mirror_pos(pos))
    
    mid = B.MIRROR_MAP[pid, 0]
    mgid = np.where(B.GID[:] == mid)[0][0]
    assert mgid == 2

    old_pos = B.P[pid, :]
    old_mpos = B.P[mgid, :]

    print("-"*60)
    print(pid, mgid)
    print("-"*60)
    print(B.P[:N2,:])
    print(B.Q[:N2,:])
    print("~"*60)
    old_aa = _a_a()
    old_aa2 = _a_a2()
    print("pos", pos, "mpos", mpos)


    U0 = _u0(0) # U0 is ab + bb for ix==0
    abp_bbp = _direct3((pos, mpos))
    bb   = _b_b()
    bpbp = _bp_bp((pos,     mpos    ))
    bbp  = _b_bp((pos,     mpos    ), (old_pos, old_mpos))
    

    print("U0", U0)
    print("phi_0\t\t", phi_0)
    print("abp_bbp\t\t", abp_bbp)
    print("bb\t\t", bb)
    print("bbp\t\t", bbp)
    print("bpbp\t\t", bpbp)

    print("OLD 0-2 1/r", B.Q[0,0] * B.Q[2,0] / np.linalg.norm(B.P[0,:] - B.P[2,:]))

    print("AA_DIFF\t", _a_a_diff())
    new_aa = _a_a()

    # B has the proposed configuration here
    B.P[pid, :] = pos
    B.P[mgid, :] = mpos
    phi_1= _direct2()

    print("old_aa - new_aa", old_aa - new_aa)
    print("NEW 0-2 1/r", B.Q[0,0] * B.Q[2,0] / np.linalg.norm(pos - mpos))
    print("phi_1\t\t", phi_1)
    print("AA\t", old_aa)
    print("P0 - 2*U0 + BB - AA\t", phi_0 - 2*U0 - old_aa + bb)

    # print("P0 - 2*U0 + BB + 2*abp_bbp - 2*bbp:", 2*phi_0 - 2*U0 + 2*abp_bbp - 2*bbp)
    attempt = phi_0 - 2*U0 + bb + 2*abp_bbp - 2*bbp + bpbp
    print("P0 - 2*U0 + BB + 2*abp_bbp - 2*bbp + bpbp:", attempt)
    print("ERR = attempt - phi_1 = ", abs(attempt - phi_1))


    B.P[pid, :] = old_pos
    B.P[mgid, :] = old_mpos

    for rx in range(N):
        pid = rng.randint(0, N-1)
        pid = 0

        
        mid = B.MIRROR_MAP[pid, 0]
        mgid = np.where(B.GID[:] == mid)[0][0]

        U0 = _u0(0) # U0 is ab + bb for ix==0
        abp_bbp = _direct3((pos, mpos))
        bb   = _b_b()
        bpbp = _bp_bp((pos,     mpos    ))
        bbp  = _b_bp((pos,     mpos    ), (old_pos, old_mpos))
        ss   = 2*bbp - bpbp - bb

        old_pos = B.P[pid, :]
        old_mpos = B.P[mgid, :]
        B.P[pid, :] = pos
        B.P[mgid, :] = mpos

        phi_1 = _direct2()

        B.P[pid, :] = old_pos
        B.P[mgid, :] = old_mpos

        print("python U0", U0)
        print("python phi_0\t\t", phi_0)
        print("python abp_bbp\t\t", abp_bbp)
        print("python bb\t\t", bb)
        print("python bbp\t\t", bbp)
        print("python bpbp\t\t", bpbp)
        print("python ss\t\t", ss)
        print("python ss = 2*bbp - bpbp - bb")


        attempt = phi_0 - 2*U0 + 2*abp_bbp - ss
        print("python err:", abs(attempt - phi_1))


        prop_energy = kmc_fmm.test_propose(
            moves=((pid, pos),)
        )
        
        assert abs(phi_direct) > 0
        err = abs(prop_energy[0][0] - phi_1)
        print("ERR", err, prop_energy[0][0], phi_1)
        return
        assert err < 10.**-16

    fmm.free()
    kmc_fmm.free()

direction_bools = (
    (True, False, False),
    (False, True, False),
    (False, False, True)
)

@pytest.mark.parametrize("direction", direction_bools)
@pytest.mark.skipif('MPISIZE > 1')
def test_kmc_fmm_nearest_27_1(direction):
    """
    Tests proposed moves one by one against direct calculation.
    Considers the primary image and the nearest 27 neighbours.
    """

    eps = 10.**-5
    L = 12
    R = 3

    N = 20
    N2 = 2 * N
    E = 4.
    rc = E/4


    rng = np.random.RandomState(seed=1234)

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()
    A.npart = N2

    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)
    A.TMP_POTENTIAL = data.ParticleDat(ncomp=1)
    A.GID = data.ParticleDat(ncomp=1, dtype=INT64)
    A.MIRROR_MAP = data.ParticleDat(ncomp=2, dtype=INT64)


    B = state.State()
    B.domain = domain.BaseDomainHalo(extent=(E,E,E))
    B.domain.boundary_condition = domain.BoundaryTypePeriodic()
    B.npart = N2

    B.P = data.PositionDat(ncomp=3)
    B.Q = data.ParticleDat(ncomp=1)
    B.GID = data.ParticleDat(ncomp=1, dtype=INT64)
    B.MIRROR_MAP = data.ParticleDat(ncomp=2, dtype=INT64)


    half_extent = [E/2 if bx else E for bx in direction]

    S = state.State()
    S.npart = N
    S.domain = domain.BaseDomainHalo(extent=half_extent)
    S.domain.boundary_condition = domain.BoundaryTypePeriodic()
    S.P = data.PositionDat()
    S.Q = data.ParticleDat(ncomp=1)
    S.GID = data.ParticleDat(ncomp=1, dtype=ctypes.c_int64)

    for dimx in range(3):
        he = 0.25*E if direction[dimx] else 0.5*E
        S.P[:, dimx] = rng.uniform(low=-he, high=he, size=N)

    S.Q[:,0] = rng.uniform(size=N)

    S.GID[:N, 0] = np.arange(N)

    MCS = kmc_dirichlet_boundary.MirrorChargeSystem(direction, S, 'P', 'Q', 'GID')
    MS = MCS.mirror_state
    
    A.P[:N2, :] = MS.P[:N2, :]
    A.Q[:N2, :] = MS.Q[:N2, :]
    A.GID[:N2, :] = MS.GID[:N2, :]
    A.MIRROR_MAP[:N2, :]  = MS.mirror_map[:N2, :]
    
    B.P[:N2, :] = MS.P[:N2, :]
    B.Q[:N2, :] = MS.Q[:N2, :]
    B.GID[:N2, :] = MS.GID[:N2, :]
    B.MIRROR_MAP[:N2, :]  = MS.mirror_map[:N2, :]

    A.scatter_data_from(0)
    B.scatter_data_from(0)

    fmm_bc = '27'
    kmc_bc = '27'

    kmc_fmm = KMCFMM(positions=A.P, charges=A.Q, domain=A.domain, r=R, l=L, 
        boundary_condition=kmc_bc, mirror_direction=direction)
    kmc_fmm.initialise()
    fmm = PyFMM(B.domain, N=N2, free_space=fmm_bc, r=kmc_fmm.fmm.R, l=kmc_fmm.fmm.L)

    def _direct():
        _phi_direct = fmm(positions=B.P, charges=B.Q)
        return _phi_direct
    
    def _mirror_pos(rpos):
        f = [-1.0 if dx else 1.0 for dx in direction]
        return (rpos[0] * f[0], rpos[1] * f[1], rpos[2] * f[2])

    def _make_prop_pos():
        p = [0,0,0]
        for dimx in range(3):
            he = 0 if direction[dimx] else 0.5*E
            p[dimx] = rng.uniform(low=-0.5*E, high=he, size=1)[0]
        return p


    for rx in range(N):
        pid = rng.randint(0, N-1)
        mid = B.MIRROR_MAP[pid, 0]
        mgid = np.where(B.GID[:] == mid)[0][0]

        pos = _make_prop_pos()
        mpos = np.array(_mirror_pos(pos))

        old_pos = B.P[pid, :]
        old_mpos = B.P[mgid, :]
        B.P[pid, :] = pos
        B.P[mgid, :] = mpos

        phi_1 = _direct()

        B.P[pid, :] = old_pos
        B.P[mgid, :] = old_mpos

        prop_energy = kmc_fmm.test_propose(
            moves=((pid, pos),)
        )
        
        assert abs(phi_1) > 0
        err = abs(prop_energy[0][0] - phi_1) / abs(phi_1)
        #print("ERR", err, prop_energy[0][0], phi_1)
        assert err < 10.**-5

    fmm.free()
    kmc_fmm.free()

@pytest.mark.parametrize("direction", direction_bools)
@pytest.mark.skipif('MPISIZE > 1')
def test_kmc_fmm_nearest_27_2(direction):
    """
    Tests proposed moves one by one against direct calculation.
    Considers the primary image and the nearest 27 neighbours.
    """

    eps = 10.**-5
    L = 12
    R = 3

    N = 20
    N2 = 2 * N
    E = 4.
    rc = E/4
    M = 8


    rng = np.random.RandomState(seed=8372)

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()
    A.npart = N2

    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)
    A.GID = data.ParticleDat(ncomp=1, dtype=INT64)
    A.MIRROR_MAP = data.ParticleDat(ncomp=2, dtype=INT64)
    A.prop_masks = data.ParticleDat(ncomp=M, dtype=INT64)
    A.prop_positions = data.ParticleDat(ncomp=M*3)
    A.prop_diffs = data.ParticleDat(ncomp=M)
    A.sites = data.ParticleDat(ncomp=1, dtype=INT64)

    site_max_counts = data.ScalarArray(ncomp=8, dtype=INT64)
    site_max_counts[:] = rng.randint(0, 10, size=8)

    B = state.State()
    B.domain = domain.BaseDomainHalo(extent=(E,E,E))
    B.domain.boundary_condition = domain.BoundaryTypePeriodic()
    B.npart = N2

    B.P = data.PositionDat(ncomp=3)
    B.Q = data.ParticleDat(ncomp=1)
    B.GID = data.ParticleDat(ncomp=1, dtype=INT64)
    B.MIRROR_MAP = data.ParticleDat(ncomp=2, dtype=INT64)


    half_extent = [E/2 if bx else E for bx in direction]

    S = state.State()
    S.npart = N
    S.domain = domain.BaseDomainHalo(extent=half_extent)
    S.domain.boundary_condition = domain.BoundaryTypePeriodic()
    S.P = data.PositionDat()
    S.Q = data.ParticleDat(ncomp=1)
    S.GID = data.ParticleDat(ncomp=1, dtype=ctypes.c_int64)

    for dimx in range(3):
        he = 0.25*E if direction[dimx] else 0.5*E
        S.P[:, dimx] = rng.uniform(low=-he, high=he, size=N)

    S.Q[:,0] = rng.uniform(size=N)

    S.GID[:N, 0] = np.arange(N)

    MCS = kmc_dirichlet_boundary.MirrorChargeSystem(direction, S, 'P', 'Q', 'GID')
    MS = MCS.mirror_state
    
    A.P[:N2, :] = MS.P[:N2, :]
    A.Q[:N2, :] = MS.Q[:N2, :]
    A.GID[:N2, :] = MS.GID[:N2, :]
    A.MIRROR_MAP[:N2, :]  = MS.mirror_map[:N2, :]
    A.sites[:, 0] = rng.randint(0, 8, size=N2)

    B.P[:N2, :] = MS.P[:N2, :]
    B.Q[:N2, :] = MS.Q[:N2, :]
    B.GID[:N2, :] = MS.GID[:N2, :]
    B.MIRROR_MAP[:N2, :]  = MS.mirror_map[:N2, :]

    A.scatter_data_from(0)
    B.scatter_data_from(0)

    fmm_bc = '27'
    kmc_bc = '27'

    kmc_fmmA = KMCFMM(positions=A.P, charges=A.Q, domain=A.domain, r=R, l=L, 
        boundary_condition=kmc_bc, mirror_direction=direction)
    kmc_fmmA.initialise()

    kmc_fmmB = KMCFMM(positions=B.P, charges=B.Q, domain=B.domain, r=R, l=L, 
        boundary_condition=kmc_bc, mirror_direction=direction)
    kmc_fmmB.initialise()
    
    for testx in range(4):
        prop = []
        nmov = 0
        for px in range(N):
            tmp = []
            masks = np.zeros(M)
            masks[:site_max_counts[A.sites[px,0]]:] = 1
            masks = rng.permutation(masks)

            for propx in range(M):
                mask = masks[propx]
                prop_pos = rng.uniform(low=-0.5*E, high=0.5*E, size=3)
                A.prop_masks[px, propx] = mask
                A.prop_positions[px, propx*3:propx*3+3:] = prop_pos
                
                if mask > 0:
                    tmp.append(list(prop_pos))
                    nmov += 1
            if len(tmp) > 0:
                prop.append((px, np.array(tmp)))

        t0 = time.time()
        correct = kmc_fmmB.propose(prop)
        t1= time.time()
        to_test =  kmc_fmmA.propose_with_dats(site_max_counts, A.sites,
            A.prop_positions, A.prop_masks, A.prop_diffs, diff=True)
        t2 = time.time()


        for propi, propx in enumerate(prop):
            pid = propx[0]
            movs = propx[1]
            found_movs = 0
            for pmi in range(M):
                if A.prop_masks[pid, pmi] > 0:
                    correct_energy = correct[propi][found_movs]
                    to_test_energy = A.prop_diffs[pid, pmi] + kmc_fmmA.energy
                    
                    rel = 1.0 if abs(correct_energy) < 1 else abs(correct_energy)
                    err = abs(correct_energy - to_test_energy) / rel
                    
                    assert err < 2*(10**-14)

                    found_movs += 1

    kmc_fmmA.free()
    kmc_fmmB.free()


