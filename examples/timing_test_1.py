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

def time_test_1(N=100, nprop=None, nsample=10):


    eps = 10.**-5
    L = 14
    R = max(3, int(log(0.2*N, 8)))

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
    
    bcs = (False, 'pbc')
    # bcs = (False, 'free_space')
 
    kmc_fmm = KMCFMM(positions=A.P, charges=A.Q, 
        domain=A.domain, r=R, l=L, boundary_condition=bcs[1])
    kmc_fmm.initialise()

    # make  some random proposed moves
    order = rng.permutation(range(N))
    prop = []
    nm = 0
    for px in range(nsample):
        
        if nprop is None:
            propn = rng.randint(1, 8)
        else:
            propn = nprop

        nm += propn
        prop.append(
            (
                order[px],
                rng.uniform(low=-0.5*E, high=0.5*E, size=(propn, 3))
            )
        )
    
    # get the energy of the proposed moves
    t0 = time.time()
    prop_energy = kmc_fmm.test_propose(moves=prop)
    t1 = time.time()
    return (t1-t0, nm, kmc_fmm.fmm.R)

if __name__ == '__main__':
    nset = np.logspace(3, log(1000001, 10), 30)
    # nset = (1000000,)

    print('{: ^10} {: ^12} {: ^4}' .format('N', 'Time', 'R'))
    print('-' * 28)
    times = []
    for nx in nset:
        ti, ni, ri = time_test_1(N=int(nx), nprop=10)
        times.append((int(nx), ti/ni, ri))

        print('{: 8.2e} {: 8.4e} {: 4d}' .format(int(nx), ti/ni, ri))


    times = np.array(times)
    np.save('timings.npy', times)







