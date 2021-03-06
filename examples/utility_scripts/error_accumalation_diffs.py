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


from mpi4py import MPI
MPISIZE = MPI.COMM_WORLD.Get_size()
MPIRANK = MPI.COMM_WORLD.Get_rank()
MPIBARRIER = MPI.COMM_WORLD.Barrier
DEBUG = True
SHARED_MEMORY = 'omp'

from coulomb_kmc import *
import time
import cProfile

INT64 = ctypes.c_int64

import sys

    
class ErrorPropDiff:
    def __init__(self, N, L):
        R = max(3, int(log(0.5*N, 8)))

        E = (N*100.)**(1./3.)

        self.E = E
        self.N = N

        A = state.State()
        A.domain = domain.BaseDomainHalo(extent=(E,E,E))
        A.domain.boundary_condition = domain.BoundaryTypePeriodic()
        A.npart = N
        A.P = data.PositionDat(ncomp=3)
        A.Q = data.ParticleDat(ncomp=1)
        A.GID = data.ParticleDat(ncomp=1, dtype=INT64)
        A.F = data.ParticleDat(ncomp=3)

        rng = np.random.RandomState(seed=1234)


        A.P[:] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,3))
        for px in range(N):
            A.Q[px,0] = (-1.0)**(px+1)
        bias = np.sum(A.Q[:N:, 0])/N
        A.Q[:, 0] -= bias
        A.GID[:,0] = np.arange(N)


        B = state.State()
        B.domain = domain.BaseDomainHalo(extent=(E,E,E))
        B.domain.boundary_condition = domain.BoundaryTypePeriodic()
        B.npart = N
        B.P = data.PositionDat(ncomp=3)
        B.Q = data.ParticleDat(ncomp=1)
        B.GID = data.ParticleDat(ncomp=1, dtype=INT64)
        B.F = data.ParticleDat(ncomp=3)

        B.P[:] = A.P[:]
        B.Q[:] = A.Q[:]
        B.F[:] = A.F[:]
        B.GID[:] = A.GID[:]

        A.scatter_data_from(0)
        B.scatter_data_from(0)
        
        bcs = (False, 'pbc')
        # bcs = (False, 'free_space')
        
        self.kmc_fmm = KMCFMM(positions=A.P, charges=A.Q, 
            domain=A.domain, r=R, l=L, boundary_condition=bcs[1])

        self.kmc_fmm_true = KMCFMM(positions=B.P, charges=B.Q, 
            domain=B.domain, r=R, l=26, boundary_condition=bcs[1])
        
        #print("TERMS REDUCED FOR DEBUGING")

        self.A = A
        self.B = B

        self.rng = rng
        
        Nc = int(math.ceil(N**(1./3)))

        self.lattice = utility.lattice.cubic_lattice((Nc, Nc, Nc), (E, E, E))[:N, :]


        self.random_initialise()

    def free(self):
        self.kmc_fmm.free()
        self.kmc_fmm_true.free()

    
    def random_initialise(self):

        self.A.P[:self.N:, :] = self.lattice + self.rng.uniform(low=-2., high=2.0, size=(self.N,3))
        self.B.P[:self.N:, :] = self.A.P[:self.N:, :]
        self.kmc_fmm.initialise()
        self.kmc_fmm_true.initialise()
        self.U_kmc = self.kmc_fmm.energy
        self.U_kmc_true = self.kmc_fmm_true.energy
        self.kmc_fmm.initialise()

    def _wrap_into_domain(self, pos):
        for dx in range(3):
            if pos[dx] < -0.5*self.E: pos[dx] += self.E
            if pos[dx] >  0.5*self.E: pos[dx] -= self.E
        return pos
    
    
    def random_propose(self):
        pid = self.rng.randint(0, self.A.npart)

        assert self.A.GID[pid, 0] == self.B.GID[pid, 0]
        direction = self.rng.uniform(low=-1.0, high=1.0, size=3)
        direction /= np.linalg.norm(direction)
        direction *= self.rng.uniform(low=0.5, high=3.0)
        pos = self._wrap_into_domain(self.A.P[pid, :] + direction)

        self.U_kmc = self.kmc_fmm.propose(((pid, pos),))[0][0]
        self.U_kmc_true = self.kmc_fmm_true.propose(((pid, pos),))[0][0]

        self.diff_kmc = self.kmc_fmm.energy - self.U_kmc
        self.diff_kmc_true = self.kmc_fmm_true.energy - self.U_kmc_true

    def random_accept(self):
        pid = self.rng.randint(0, self.A.npart)
        direction = self.rng.uniform(low=-1.0, high=1.0, size=3)
        direction /= np.linalg.norm(direction)
        direction *= self.rng.uniform(low=0.5, high=3.0)
        pos = self._wrap_into_domain(self.A.P[pid, :] + direction)
        
        assert self.A.GID[pid, 0] == self.B.GID[pid, 0]

        # this updates the position dat
        self.kmc_fmm.accept((pid, pos))
        self.kmc_fmm_true.accept((pid, pos))

        self.diff_kmc = self.kmc_fmm.energy - self.U_kmc
        self.diff_kmc_true = self.kmc_fmm_true.energy - self.U_kmc_true

        self.U_kmc = self.kmc_fmm.energy
        self.U_kmc_true = self.kmc_fmm_true.energy   
    
    @property
    def diff_error(self):
        rel = abs(self.diff_kmc_true)
        rel = 1.0 if rel == 0.0 else rel
        return self.diff_kmc_true, abs(self.diff_kmc - self.diff_kmc_true) / rel, self.kmc_fmm_true.energy
    
    @property
    def diff_error_raw(self):
        return self.diff_kmc_true, self.diff_kmc, self.kmc_fmm_true.energy   


if __name__ == '__main__':

    N = int(sys.argv[1])
    L = int(sys.argv[2])

    proposer = ErrorPropDiff(N, L)
    print("=" * 80)
    print("N:\t\t", N)
    print("Extent:\t", proposer.E)
    print("=" * 80)
    niter = 10**4

    data = np.zeros((niter, 3), 'float64')


    for tx in range(niter):
        if (tx > 0) and (tx % 100 == 0):
            proposer.random_initialise()

        proposer.random_propose()
        e = proposer.diff_error_raw
        
        data[tx, :] = e
        
        print("{: 5d} | {: 16.8e} {: 16.8e} | {: 8.2e}".format(*([tx] + list(e))))
    
    np.save('./error_data_raw_{}_{}.npy'.format(N, L), data)
    proposer.free()




