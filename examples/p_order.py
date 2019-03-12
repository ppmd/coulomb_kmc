from __future__ import print_function, division

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"

import ctypes, math
import numpy as np

np.set_printoptions(linewidth=200)


from ppmd import *
from ppmd.coulomb.fmm import *
from ppmd.coulomb.ewald_half import *
from scipy.special import sph_harm, lpmv
import time

from math import *

MPISIZE = mpi.MPI.COMM_WORLD.Get_size()
MPIRANK = mpi.MPI.COMM_WORLD.Get_rank()
MPIBARRIER = mpi.MPI.COMM_WORLD.Barrier
DEBUG = True
SHARED_MEMORY = 'omp'

from coulomb_kmc import *
import time
import cProfile



def p_complexity(E, pi, qi, p, N=1000, nprop=2, nsample=1000):
    
    assert N >= nsample
    assert p > 0

    eps = 10.**-5
    L = int(p)
    R = max(3, int(log(0.5*N, 8)))

    rc = E/4

    M = nprop

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()
    A.npart = N
    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)
    A.prop_masks = data.ParticleDat(ncomp=M, dtype=INT64)
    A.prop_positions = data.ParticleDat(ncomp=M*3)
    A.prop_diffs = data.ParticleDat(ncomp=M)
    A.sites = data.ParticleDat(ncomp=1, dtype=INT64)

    rng = np.random.RandomState(seed=1234)

    site_max_counts = data.ScalarArray(ncomp=8, dtype=INT64)
    site_max_counts[:] = nprop

    A.P[:] = pi
    A.Q[:, 0] = qi
    
    A.scatter_data_from(0)
    
    bcs = (False, 'pbc')
    # bcs = (False, 'free_space')
 
    kmc_fmm = KMCFMM(positions=A.P, charges=A.Q, 
        domain=A.domain, r=R, l=L, boundary_condition=bcs[1])
    kmc_fmm.initialise()

    # make  some random proposed moves
    order = rng.permutation(range(N))
    nm = 0

    for px in range(nsample):
        mp = site_max_counts[A.sites[px,0]]
        for propx in range(mp):
            nm += 1
            prop_pos = rng.uniform(low=-0.5*E, high=0.5*E, size=3)
            A.prop_masks[px, propx] = 1
            A.prop_positions[px, propx*3:propx*3+3:] = prop_pos


    pr = cProfile.Profile()
    pr.enable()
    t0 = time.time()
    to_test =  kmc_fmm.propose_with_dats(site_max_counts, A.sites,
        A.prop_positions, A.prop_masks, A.prop_diffs, diff=True)
    t1 = time.time()
    pr.disable()
    pr.dump_stats('/tmp/propose.prof')
 
    pr = cProfile.Profile()
    pr.enable()   

    naccept = 0
    nsample2 = 10
    t2 = time.time()

    for px in range(N):
        if naccept == nsample2:
            break
        for propx in range(M):
            kmc_fmm.accept((px, A.prop_positions[px, propx*3: (propx+1)*3:]))
            naccept+=1
            if naccept == nsample2:
                break

    t3 = time.time()
    pr.disable()
    pr.dump_stats('/tmp/accept.prof')

    kmc_fmm.free()

    return (t1-t0, nm, kmc_fmm.fmm.R, t3 - t2, nsample2)

if __name__ == '__main__':
    pset = tuple(range(2, 31, 1))
    N = 100000
    E = 3.3 * (N ** (1./3))

    rng = np.random.RandomState(1234)
    pi = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,3))
    qi = np.zeros(N)
    for px in range(N):
        qi[px] = (-1.0)**(px+1)
    bias = np.sum(qi[:N:])/N
    qi[:] -= bias


    top_bar = '{: ^4} {: ^12} {: ^12} {: ^4}' .format('N', 'T_prop', 'T_accept', 'R')
    print(top_bar)
    print('-' * len(top_bar))
    times = []
    for px in pset:
        ti, ni, ri, tai, nsample2 = p_complexity(E, pi, qi, p=px, N=N, nprop=10, nsample=1000)
        times.append((px, ti/ni, tai/(nsample2*N)))

        print('{: 4d} {: 8.4e} {: 8.4e} {: 4d}' .format(int(px), ti/ni, tai/(nsample2*N), ri))


    times = np.array(times)
    np.save('porder_timings.npy', times)



