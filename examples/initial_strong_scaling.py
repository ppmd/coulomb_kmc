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


ParticleLoop = loop.ParticleLoopOMP
ParticleDat = data.ParticleDat
PositionDat = data.PositionDat
ScalarArray = data.ScalarArray
PairLoop = pairloop.CellByCellOMP


Constant = kernel.Constant
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



# setup the state
A = state.State()
A.domain = domain.BaseDomainHalo(extent=(E,E,E))
A.domain.boundary_condition = domain.BoundaryTypePeriodic()
A.npart = N
A.P = PositionDat(ncomp=3)
A.Q = ParticleDat(ncomp=1)
A.prop_masks = ParticleDat(ncomp=M, dtype=INT64)
A.prop_positions = ParticleDat(ncomp=M*3)
A.prop_diffs = ParticleDat(ncomp=M)
A.sites = ParticleDat(ncomp=1, dtype=INT64)

site_max_counts = ScalarArray(ncomp=1, dtype=INT64)
site_max_counts[:] = M

offsets_sa = ScalarArray(ncomp=offsets_array.shape[0], dtype=REAL)
offsets_sa[:] = offsets_array.copy()

# load data into dats
A.P[:] = loaded_data['P'].copy()
A.Q[:, 0] = loaded_data['Q'].copy()
A.sites[:, 0] = 0
A.scatter_data_from(0)


# create kmc instance
kmc_fmm = KMCFMM(positions=A.P, charges=A.Q, domain=A.domain, r=R, l=L,
    boundary_condition='pbc', max_move=max_move_dim)
kmc_fmm.initialise()




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
        'P': A.P(READ),
        'PP': A.prop_positions(WRITE),
        'OA': offsets_sa(READ),
        'MASK': A.prop_masks(WRITE)
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
    MASK.i[mx] = (r2 < TOL) ? 0 :  MASK.i[mx];
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
        'P': A.P(READ),
        'OA': offsets_sa(READ),
        'MASK': A.prop_masks(WRITE)
    },
    shell_cutoff = max_move
)




MPIBARRIER()
t0 = time.time()

for stepx in range(num_steps):
    
    prop_pos.execute()
    exclude.execute()

    kmc_fmm.propose_with_dats(site_max_counts, A.sites,
        A.prop_positions, A.prop_masks, A.prop_diffs, diff=True)






MPIBARRIER()
t1 = time.time()


if MPIRANK == 0:

    print('-' * 80)
    print("Time taken: \t", t1 - t0)
    print("N:\t\t", N)
    print("M:\t\t", M)
    print("NSTEP:\t\t", num_steps)

    print('-' * 80)
    opt.print_profile()
    print('-' * 80)




kmc_fmm.free()



