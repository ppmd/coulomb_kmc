from __future__ import print_function, division

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"

import ctypes
import math
import numpy as np
import itertools
import time
import pickle
import sys
from itertools import product

from ppmd import *
from ppmd.coulomb.fmm import *
from ppmd.coulomb.ewald_half import *
from scipy.special import sph_harm, lpmv

MPISIZE = mpi.MPI.COMM_WORLD.Get_size()
MPIRANK = mpi.MPI.COMM_WORLD.Get_rank()
MPIBARRIER = mpi.MPI.COMM_WORLD.Barrier
NTHREADS = runtime.NUM_THREADS


ParticleDat = data.ParticleDat
PositionDat = data.PositionDat
ScalarArray = data.ScalarArray
GlobalArray = data.GlobalArray

ParticleLoop = loop.ParticleLoopOMP
PairLoop = pairloop.CellByCellOMP


Constant = kernel.Constant
Header = kernel.Header

from ppmd.access import *
from coulomb_kmc import *


# lattice creation each site same type
stencil = (-2, -1, 0, 1, 2)
def _ox_len(ox):
    # remove the no offset from the stencil
    if (ox[0] == 0) and (ox[1] == 0) and (ox[2] == 0): return False
    
    # first shell
    one_norm = abs(ox[0]) + abs(ox[1]) + abs(ox[2])
    if one_norm == 1: return True
    
    # outer corners
    if (abs(ox[0]) == 2) and (abs(ox[1]) == 2) and (abs(ox[2]) == 2): return True

    return False


offsets = [ox for ox in product(stencil, stencil, stencil) if _ox_len(ox)]
offsets_matrix = np.array(offsets, REAL)

# convert to real valued lattice points from offset indices
offsets_matrix *= 1.1

offsets_array = offsets_matrix.ravel()

max_move = 0
for ox in range(offsets_matrix.shape[0]):
    max_move = max(max_move, np.linalg.norm(offsets_matrix[ox, :]))
max_move *= 1.05
max_move_dim = 1.05 * np.max(np.abs(offsets_array))


# parameters
filename = sys.argv[1]
with open(filename, 'rb') as fh:
    loaded_data = pickle.load(fh)


num_steps = int(sys.argv[2])
N = loaded_data['P'].shape[0]
L = 12
R = max(3, int(log(0.5*N, 8)))
E = loaded_data['E']
rng = np.random.RandomState(seed=1234)
M = offsets_matrix.shape[0]


if MPIRANK == 0:

    print('-' * 80)
    print("N:\t", N)
    print("R:\t", R)
    print("L:\t", L)
    print("E:\t", E)
    print("M:\t", M)


# setup the state
A = state.State()
A.domain = domain.BaseDomainHalo(extent=(E,E,E))
A.domain.boundary_condition = domain.BoundaryTypePeriodic()
A.npart = N
A.P = PositionDat(ncomp=3)
A.Q = ParticleDat(ncomp=1)
A.GID = ParticleDat(ncomp=1, dtype=INT64)
A.prop_masks = ParticleDat(ncomp=M, dtype=INT64)
A.prop_rates = ParticleDat(ncomp=M, dtype=REAL)
A.prop_rate_totals = ParticleDat(ncomp=1, dtype=REAL)
A.prop_positions = ParticleDat(ncomp=M*3)
A.prop_diffs = ParticleDat(ncomp=M)
A.prop_inc_sum = ParticleDat(ncomp=M) 

A.sites = ParticleDat(ncomp=1, dtype=INT64)

site_max_counts = ScalarArray(ncomp=1, dtype=INT64)
site_max_counts[:] = M

offsets_sa = ScalarArray(ncomp=offsets_array.shape[0], dtype=REAL)
offsets_sa[:] = offsets_array.copy()

# load data into dats
A.P[:] = loaded_data['P'].copy()
A.Q[:, 0] = loaded_data['Q'].copy()
A.sites[:, 0] = 0
A.GID[:, 0] = np.arange(N)
A.scatter_data_from(0)


# create kmc instance
kmc_fmm = KMCFMM(positions=A.P, charges=A.Q, domain=A.domain, r=R, l=L,
    boundary_condition='pbc', max_move=max_move_dim)
kmc_fmm.initialise()

if MPIRANK == 0:
    print('-' * 80)
    opt.print_profile()
    print('-' * 80)


# make proposed positions kernel
prop_pos_kernel_src = r'''
// current pos
const double p0 = P.i[0];
const double p1 = P.i[1];
const double p2 = P.i[2];

// reset mask
for (int mx=0 ; mx<M ; mx++){
    MASK.i[mx] = 1;
}

// form proposed positions
for (int mx=0 ; mx< M ; mx++){
    double n0 = p0 + OA[mx*3 + 0];
    double n1 = p1 + OA[mx*3 + 1];
    double n2 = p2 + OA[mx*3 + 2];
    
    if ( n0 < LOWER ) { n0 += EXTENT; } 
    if ( n1 < LOWER ) { n1 += EXTENT; } 
    if ( n2 < LOWER ) { n2 += EXTENT; } 

    if ( n0 > UPPER ) { n0 -= EXTENT; } 
    if ( n1 > UPPER ) { n1 -= EXTENT; } 
    if ( n2 > UPPER ) { n2 -= EXTENT; } 

    PP.i[mx*3 + 0] = n0;
    PP.i[mx*3 + 1] = n1;
    PP.i[mx*3 + 2] = n2;
}
'''
prop_pos_kernel = kernel.Kernel(
    'prop_pos_kernel', 
    prop_pos_kernel_src, 
    constants=(
        Constant('M', M),
        Constant('LOWER', -0.5 * E),
        Constant('UPPER', 0.5 * E),
        Constant('EXTENT', E)
    )
)
prop_pos = ParticleLoop(
    kernel=prop_pos_kernel, 
    dat_dict={
        'P'     : A.P(READ),
        'PP'    : A.prop_positions(WRITE),
        'OA'    : offsets_sa(READ),
        'MASK'  : A.prop_masks(WRITE)
    }
)


# make exclude kernel

exclude_kernel_src = r'''
// current pos
const double p0 = P.i[0];
const double p1 = P.i[1];
const double p2 = P.i[2];

// j position
const double pj0 = P.j[0];
const double pj1 = P.j[1];
const double pj2 = P.j[2];

// check each proposed position
for (int mx=0 ; mx< M ; mx++){
    double n0 = p0 + OA[mx*3 + 0];
    double n1 = p1 + OA[mx*3 + 1];
    double n2 = p2 + OA[mx*3 + 2];

    const double d0 = pj0 - n0;
    const double d1 = pj1 - n1;
    const double d2 = pj2 - n2;
    
    const double r2 = d0*d0 + d1*d1 + d2*d2;
    MASK.i[mx] = (r2 < TOL) ? 0 : MASK.i[mx];
}
'''
exclude_kernel = kernel.Kernel(
    'exclude_kernel', 
    exclude_kernel_src, 
    constants=(
        Constant('M', M),
        Constant('TOL', 0.01)
    )
)
exclude = PairLoop(
    kernel=exclude_kernel, 
    dat_dict={
        'P'     : A.P(READ),
        'OA'    : offsets_sa(READ),
        'MASK'  : A.prop_masks(WRITE)
    },
    shell_cutoff = max_move
)

# make rate kernel

rate_kernel_src = r'''
double charge_rate = 0.0;
for (int mx=0 ; mx< M ; mx++){
    const double rate = (MASK.i[mx] > 0) ? exp( -1.0 * DU.i[mx] ) : 0.0 ;
    R.i[mx] = rate;
    charge_rate += rate;
    PIS.i[mx] = charge_rate;
}

RT.i[0] = charge_rate;
'''

rate_kernel = kernel.Kernel(
    'rate_kernel',
    rate_kernel_src,
    constants=(
        Constant('M', M),
    ),
    headers=(Header('math.h'),)
)

rate = ParticleLoop(
    kernel=rate_kernel,
    dat_dict={
        'MASK'  : A.prop_masks(READ),
        'DU'    : A.prop_diffs(READ),
        'R'     : A.prop_rates(WRITE),
        'RT'    : A.prop_rate_totals(WRITE),
        'PIS'   : A.prop_inc_sum(WRITE)
    }
)

move_logic_time = 0.0

def find_charge_to_move():
    mt0 = time.time()
    
    # compute rates from differences
    rate.execute()
    
    
    local_inc_sum = np.cumsum(A.prop_rate_totals[:A.npart_local:, 0])
    local_total = np.array(local_inc_sum[-1], REAL)


    all_totals = np.zeros(MPISIZE, REAL)
    A.domain.comm.Allgather(local_total, all_totals)
    all_inc_sum = np.cumsum(all_totals)
    
    rate_total = all_inc_sum[-1]

    accept_point = rng.uniform(0.0, rate_total - 4*10.**-16)
    
    if MPIRANK == 0:
        low = 0.0
    else:
        low = all_inc_sum[MPIRANK - 1]
    
    high = all_inc_sum[MPIRANK]


    if (accept_point >= low ) and (accept_point < high):
        # this rank accepts
        if A.npart_local == 0: raise RuntimeError('ERROR: this rank accepted but has no charges.')
        
        # offset accept point onto MPI rank
        accept_point_local = accept_point - low

        m = np.searchsorted(local_inc_sum, accept_point_local)
        assert m >= 0
        assert m < A.npart_local

        
        if m == 0:
            charge_low = low
        else:
            charge_low = local_inc_sum[m - 1]



        accept_point_charge = accept_point_local - charge_low


        e = np.searchsorted(A.prop_inc_sum.view[m, :], accept_point_charge)

        assert e >= 0
        assert e < M
        assert A.prop_masks[m, e] > 0
        
        move = (m, A.prop_positions[m, e*3:(e+1)*3:].copy())

    else:
        # rank does not accept
        move = None

    kmc_fmm.accept(move)
    global move_logic_time
    move_logic_time += time.time() - mt0


print_str = r'{: 7d} | {: 20.16e}'

PRINT = False


if MPIRANK == 0:
    print('-' * 80)
    print('{:7s} | {:20s}'.format('step', 'energy'))
    print('-' * 80)
    print(print_str.format(-1, kmc_fmm.energy))


MPIBARRIER()
t0 = time.time()

for stepx in range(num_steps):
    
    prop_pos.execute()
    exclude.execute()

    kmc_fmm.propose_with_dats(site_max_counts, A.sites,
        A.prop_positions, A.prop_masks, A.prop_diffs, diff=True)
    
    find_charge_to_move()

    if MPIRANK == 0 and PRINT:
        print(print_str.format(stepx, kmc_fmm.energy))


MPIBARRIER()
t1 = time.time()


if MPIRANK == 0:
    print(print_str.format(stepx, kmc_fmm.energy))


if MPIRANK == 0:
    time_taken = t1 - t0
    print('-' * 80)
    print("Time taken: \t", time_taken)
    print("Time in accept:\t", move_logic_time)
    print("N:\t\t\t\t", N)
    print("M:\t\t\t\t", M)
    print("NSTEP:\t\t\t", num_steps)

    print('-' * 80)
    opt.print_profile()
    print('-' * 80)

    with open('./timing_{}_{}_{}_{}.result'.format(MPISIZE, NTHREADS, N, num_steps), 'a') as fh:
        fh.write(str(time_taken))

kmc_fmm.free()



