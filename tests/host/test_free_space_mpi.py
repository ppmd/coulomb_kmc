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


from kmc_test_common import *


@pytest.mark.parametrize("R", (3, 4, 5))
def test_kmc_fmm_free_space_1(R):
    """
    Tests proposed moves one by one against direct calculation.
    """

    if R < 4 and MPISIZE > 8:
        return

    eps = 10.**-5
    L = 12

    N = 100
    E = 2*3.1415
    rc = E/4

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()
    A.npart = N

    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)
    A.GID = data.ParticleDat(ncomp=1, dtype=INT64)

    A.crr = data.ScalarArray(ncomp=1)

    rng = np.random.RandomState(seed=9184)

    pi = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,3))
    ppi = pi.copy()

    qi = np.zeros((N, 1))
    for px in range(N):
        qi[px,0] = (-1.0)**(px+1)
    bias = np.sum(qi[:N:, 0])/N
    qi -= bias


    A.P[:] = pi
    A.Q[:] = qi
    A.GID[:, 0] = np.arange(N)

    A.scatter_data_from(0)
    

    FSD = FreeSpaceDirect()

    def _direct():
        
        _phi_c = FSD(N, ppi, qi)
        return _phi_c

        _phi_direct = 0.0
        # compute phi from image and surrounding 26 cells
        for ix in range(N):
            for jx in range(ix+1, N):
                rij = np.linalg.norm(ppi[jx,:] - ppi[ix,:])
                _phi_direct += qi[ix, 0] * qi[jx, 0] / rij

        return _phi_direct
    
    phi_direct = _direct()


    kmc_fmm = KMCFMM(positions=A.P, charges=A.Q, 
        domain=A.domain, r=R, l=L, boundary_condition='free_space')

    kmc_fmm.initialise()


    for rx in range(200):
    #for rx in range(1):
        
        # using randint breaks seed sync between ranks
        pid = int(rng.uniform(0, 1) * A.npart_local)
        
        #pid = np.where(A.GID.view == 19)

        #if len(pid[0]) > 0:
        #    pid = pid[0][0]

        gid = A.GID[pid, 0]
        pos = rng.uniform(low=-0.5*E, high=0.5*E, size=3)
        
        #print(pid, gid, pos, MPIRANK)

        old_pos = ppi[gid, :]
        ppi[gid, :] = pos

        phi_direct = _direct()

        ppi[gid, :] = old_pos


        prop_energy = kmc_fmm.propose(
            moves=((pid, pos),)
        )
        
        #continue
        assert abs(phi_direct) > 0

        # print(prop_energy[0][0], phi_direct)
        assert (abs(prop_energy[0][0] - phi_direct)/abs(phi_direct) < eps) or (abs(prop_energy[0][0] - phi_direct) < eps)

    kmc_fmm.free()

















