from __future__ import print_function, division


__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"

import sys
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

#import psutil
#p = psutil.Process()


def time_test_dats_1(N=1000, nprop=2, nsample=20000):
    
    nsample = min(N, nsample)
    assert N >= nsample

    eps = 10.**-5
    L = 4
    R = max(3, int(log(0.5*N, 8)))
    
    E = 3.3 * (N ** (1./3))
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

    A.P[:] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,3))
    for px in range(N):
        A.Q[px,0] = (-1.0)**(px+1)
    bias = np.sum(A.Q[:N:, 0])/N
    A.Q[:, 0] -= bias
    
    A.scatter_data_from(0)
    
    bcs = (False, 'pbc')
    bcs = (False, 'free_space')
    
    max_move = E/10.

    kmc_fmm = KMCFMM(positions=A.P, charges=A.Q, 
        domain=A.domain, r=R, l=L, boundary_condition=bcs[1], max_move=max_move)


    kmc_fmm.initialise()
    
    #MPIBARRIER()
    #print("---------- del call in script", N, MPIRANK)
    #del kmc_fmm
    #print("========== del call in script", N, MPIRANK)
    #MPIBARRIER()

    #if MPIRANK == 0:
    #    print("returning early"); 
    #return (0, 1, 3, 0, 1)



    #print("-" * 80)
    #del kmc_fmm
    #print("=" * 80)
    #
    
    # make  some random proposed moves
    order = rng.permutation(range(A.npart_local))
    nm = 0
    nsample = min(nsample, A.npart_local)
    for px in range(nsample):
        mp = site_max_counts[A.sites[px,0]]
        for propx in range(mp):
            nm += 1
            direction = rng.uniform(low=-1.0, high=1.0, size=3)
            direction /= np.linalg.norm(direction)
            direction *= max_move * rng.uniform(0, 1)
            prop_pos = A.P.view[px, :] + direction
            A.prop_masks[px, propx] = 1
            A.prop_positions[px, propx*3:propx*3+3:] = prop_pos

    pr = cProfile.Profile()
    pr.enable()
    to_test =  kmc_fmm.propose_with_dats(site_max_counts, A.sites,
        A.prop_positions, A.prop_masks, A.prop_diffs, diff=True)
    pr.disable()
    pr.dump_stats('/tmp/propose.prof')


    # make  some random proposed moves
    order = rng.permutation(range(A.npart_local))
    nm = 0
    nsample = min(nsample, A.npart_local)
    for px in range(nsample):
        mp = site_max_counts[A.sites[px,0]]
        for propx in range(mp):
            nm += 1
            direction = rng.uniform(low=-1.0, high=1.0, size=3)
            direction /= np.linalg.norm(direction)
            direction *= max_move * rng.uniform(0, 1)
            prop_pos = A.P.view[px, :] + direction
            A.prop_masks[px, propx] = 1
            A.prop_positions[px, propx*3:propx*3+3:] = prop_pos


    A.domain.comm.Barrier()
    t0 = time.time()
    to_test =  kmc_fmm.propose_with_dats(site_max_counts, A.sites,
        A.prop_positions, A.prop_masks, A.prop_diffs, diff=True)
    A.domain.comm.Barrier()
    t1 = time.time()
 

    pr = cProfile.Profile()
    pr.enable()   

    naccept = 0
    nsample2 = 2

    for px in range(N):
        if naccept == nsample2:
            break
        for propx in range(M):
            # MPI ACCEPT NOT IMPLEMENTED
            #kmc_fmm.accept((px, A.prop_positions[px, propx*3: (propx+1)*3:]))
            naccept+=1
            if naccept == nsample2:
                break

    pr.disable()
    pr.dump_stats('/tmp/accept.prof')



    naccept = 0
    nsample2 = 10

    A.domain.comm.Barrier()
    t2 = time.time()
    
    for px in range(nsample2, N):
        if naccept == nsample2:
            break
        for propx in range(M):
            # MPI ACCEPT NOT IMPLEMENTED
            #kmc_fmm.accept((px, A.prop_positions[px, propx*3: (propx+1)*3:]))
            naccept+=1
            if naccept == nsample2:
                break

    A.domain.comm.Barrier()
    t3 = time.time()

    
    recv = np.array((0,))
    A.domain.comm.Reduce(np.array((nm,)), recv)
    nm = recv[0]
    

    return (t1-t0, nm, kmc_fmm.fmm.R, t3 - t2, nsample2)

if __name__ == '__main__':
    #nset = np.logspace(3, log(1000001, 10), 30)
    #nset = (1000,)
    nset = tuple([int(ix) for ix in sys.argv[1:]])
    
    top_bar = '{: ^10} {: ^12} {: ^12} {: ^4}' .format('N', 'T_prop', 'T_accept', 'R')
    if MPIRANK == 0:
        print(top_bar)
        print('-' * len(top_bar))

    times = []
    for nx in nset:
        ti, ni, ri, tai, nsample2 = time_test_dats_1(N=int(nx), nprop=10)
        # opt.print_profile()
        if MPIRANK == 0:
            times.append((int(nx), ti/ni, tai/(nsample2*nx)))
            print('{: 8.2e} {: 8.4e} {: 8.4e} {: 4d}' .format(int(nx), ti/ni, tai/(nsample2*nx), ri))

    if MPIRANK == 0:
        times = np.array(times)
        np.save('timings.npy', times)
    



