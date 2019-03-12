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

MPISIZE = MPI.COMM_WORLD.Get_size()
MPIRANK = MPI.COMM_WORLD.Get_rank()
MPIBARRIER = MPI.COMM_WORLD.Barrier
DEBUG = True
SHARED_MEMORY = 'omp'

from coulomb_kmc import *
import time
import cProfile


    
class ErrorPropExp:
    def __init__(self, N, L=12):
        R = max(3, int(log(0.2*N, 8)))

        
        E = 2. * (N**(1./3))
        rc = E/4

        A = state.State()
        A.domain = domain.BaseDomainHalo(extent=(E,E,E))
        A.domain.boundary_condition = domain.BoundaryTypePeriodic()
        A.npart = N
        A.P = data.PositionDat(ncomp=3)
        A.Q = data.ParticleDat(ncomp=1)
        A.F = data.ParticleDat(ncomp=3)

        rng = np.random.RandomState(seed=1234)


        A.P[:] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,3))
        for px in range(N):
            A.Q[px,0] = (-1.0)**(px+1)
        bias = np.sum(A.Q[:N:, 0])/N
        A.Q[:, 0] -= bias
        
        A.scatter_data_from(0)
        
        bcs = (False, 'pbc')
        # bcs = (False, 'free_space')
        
        kmc_fmm = KMCFMM(positions=A.P, charges=A.Q, 
            domain=A.domain, r=R, l=L, boundary_condition=bcs[1])
        kmc_fmm.initialise()

        self.A = A
        self.kmc_fmm = kmc_fmm
        self.fmm = PyFMM(A.domain, r=R, l=L)
        self.rng = rng
        
        self.U_kmc = kmc_fmm.energy
        self.U_fmm = self.fmm(A.P, A.Q)

    def random_accept(self):
        pid = self.rng.randint(0, self.A.npart)
        pos = self.rng.uniform(low=-0.5*self.A.domain.extent[0], high=0.5*self.A.domain.extent[0], size=3)
        
        # this updates the position dat
        self.kmc_fmm.accept((pid, pos))
    
    @property
    def energy(self):
        self.U_kmc = self.kmc_fmm.energy
        self.U_fmm = self.fmm(self.A.P, self.A.Q)
        rel = abs(self.U_kmc)
    
        return (self.U_fmm, self.U_kmc, abs(self.U_fmm - self.U_kmc) / rel)

    def free(self):
        self.kmc_fmm.free()

if __name__ == '__main__':
    
    proposer = ErrorPropExp(10000, 12)

    niter = 10**8
    for tx in range(niter):
        proposer.random_accept()

        if (tx % 100) == 0:
            e = proposer.energy
            print("{: 12d} | {: 8.2e}".format(tx, e[2]))

    proposer.free()
