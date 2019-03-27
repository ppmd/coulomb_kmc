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


def time_test_dats_1(N=1000, nprop=6, nsample=20000):
    
    nsample = min(N, nsample)
    assert N >= nsample

    eps = 10.**-5
    L = 12
    R = max(3, int(log(0.5*N, 8)))
    #R = 6
    
    E = 3.3 * (N ** (1./3))
    rc = E/4

    max_move = 4.

    M = nprop

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()
    A.npart = N
    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)
    A.GID = data.ParticleDat(ncomp=1, dtype=INT64)
    A.prop_masks = data.ParticleDat(ncomp=M, dtype=INT64)
    A.prop_positions = data.ParticleDat(ncomp=M*3)
    A.prop_diffs = data.ParticleDat(ncomp=M)
    A.sites = data.ParticleDat(ncomp=1, dtype=INT64)

    rng = np.random.RandomState(seed=1234)
    accept_rng = np.random.RandomState(seed=34372)

    site_max_counts = data.ScalarArray(ncomp=8, dtype=INT64)
    site_max_counts[:] = nprop

    A.P[:] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,3))
    for px in range(N):
        A.Q[px,0] = (-1.0)**(px+1)
    bias = np.sum(A.Q[:N:, 0])/N
    A.Q[:, 0] -= bias
    A.GID[:, 0] = np.arange(N)


    A.scatter_data_from(0)
    
    bcs = (False, 'pbc')
    



    kmc_fmm = KMCFMM(positions=A.P, charges=A.Q, 
        domain=A.domain, r=R, l=L, boundary_condition=bcs[1], max_move=max_move)
    kmc_fmm.initialise()
    

    # make  some random proposed moves

    nm = 0
    t_propose = 0.0
    
    for runx in range(4):
        order = rng.permutation(range(A.npart_local))
        nsample = min(nsample, A.npart_local)

        for px in range(nsample):
            mp = site_max_counts[A.sites[px,0]]
            for propx in range(mp):
                nm += 1
                direction = rng.uniform(low=-1.0, high=1.0, size=3)
                direction /= np.linalg.norm(direction)
                direction *= max_move * rng.uniform(0, 1)
                prop_pos = A.P.view[px, :] + direction
                for dimx in range(3):
                    prop_pos[dimx] = np.fmod(prop_pos[dimx] + 1.5*E, E) - 0.5*E

                A.prop_masks[px, propx] = 1
                A.prop_positions[px, propx*3:propx*3+3:] = prop_pos

        A.domain.comm.Barrier()
        t0 = time.time()
        to_test =  kmc_fmm.propose_with_dats(site_max_counts, A.sites,
            A.prop_positions, A.prop_masks, A.prop_diffs, diff=True)
        A.domain.comm.Barrier()
        t1 = time.time()
        t_propose += t1 - t0

 
    nsample2 = 100
    A.domain.comm.Barrier()
    t2 = time.time()
    for ax in range(nsample2):
        gid = accept_rng.randint(0, N-1)
        lid = np.where(A.GID.view[:A.npart_local:, 0] == gid)
        if len(lid[0]) > 0:
            pid = lid[0][0]

            direction = rng.uniform(low=-1.0, high=1.0, size=3)
            direction /= np.linalg.norm(direction)
            direction *= max_move * rng.uniform(0, 1)
            prop_pos = A.P.view[pid, :] + direction
            for dimx in range(3):
                prop_pos[dimx] = np.fmod(prop_pos[dimx] + 1.5*E, E) - 0.5*E

            move = (pid, prop_pos)
            kmc_fmm.accept(move)
        else:
            kmc_fmm.accept(None)

    A.domain.comm.Barrier()
    t3 = time.time()

    
    recv = np.array((0,))
    A.domain.comm.Reduce(np.array((nm,)), recv)
    nm = recv[0]
    
    kmc_fmm.free()

    return (t_propose, nm, R, t3 - t2, nsample2)

if __name__ == '__main__':

    if len(sys.argv) > 1:
        nset = tuple([int(ix) for ix in sys.argv[1:]])
    else:
        if MPISIZE < 8:
            s = 3
        elif MPISIZE < 512:
            s = 4
        else:
            s = 5

        nset = np.logspace(s, log(1000001, 10), 30)
    
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
    



