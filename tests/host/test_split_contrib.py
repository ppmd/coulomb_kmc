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

REAL = ctypes.c_double
INT64 = ctypes.c_int64


def test_split_1():
    L = 12
    R = 3

    N = 200

    E = 1.
    rc = E/4

    rng = np.random.RandomState(seed=12415)

    Near = state.State()
    Near.domain = domain.BaseDomainHalo(extent=(E,E,E))
    Near.domain.boundary_condition = domain.BoundaryTypePeriodic()
    Near.npart = N
    Near.P = data.PositionDat(ncomp=3)
    Near.Q = data.ParticleDat(ncomp=1)

    Far = state.State()
    Far.domain = domain.BaseDomainHalo(extent=(E,E,E))
    Far.domain.boundary_condition = domain.BoundaryTypePeriodic()
    Far.npart = N
    Far.P = data.PositionDat(ncomp=3)
    Far.Q = data.ParticleDat(ncomp=1)

    All = state.State()
    All.domain = domain.BaseDomainHalo(extent=(E,E,E))
    All.domain.boundary_condition = domain.BoundaryTypePeriodic()
    All.npart = N
    All.P = data.PositionDat(ncomp=3)
    All.Q = data.ParticleDat(ncomp=1)

    pi = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,3))
    qi = rng.uniform(size=(N, 1))
    qi -= np.sum(qi) / N

    with Far.modify() as mv:
        if MPIRANK == 0:
            mv.add(
                {
                    Far.P: pi,
                    Far.Q: qi
                }
            )
    with Near.modify() as mv:
        if MPIRANK == 0:
            mv.add(
                {
                    Near.P: pi,
                    Near.Q: qi
                }
            )
    with All.modify() as mv:
        if MPIRANK == 0:
            mv.add(
                {
                    All.P: pi,
                    All.Q: qi
                }
            )

    Near_kmc = kmc_fmm.KMCFMM(Near.P, Near.Q, Near.domain, boundary_condition='27', l=L, r=R)
    Far_kmc = kmc_fmm.KMCFMM(Far.P, Far.Q, Far.domain, boundary_condition='ff-only', l=L, r=R)
    All_kmc = kmc_fmm.KMCFMM(All.P, All.Q, All.domain, boundary_condition='pbc', l=L, r=R)

    Near_kmc.initialise()
    Far_kmc.initialise()
    All_kmc.initialise()
    
    # Check the initial energies match
    assert abs(All_kmc.energy) > 0.0
    err_energy = abs(Near_kmc.energy + Far_kmc.energy - All_kmc.energy) / abs(All_kmc.energy)
    assert err_energy < 10.**-14


    # Check eval_field is consistent
    eval_points = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,3))
    
    Near_eval = Near_kmc.eval_field(eval_points)
    Far_eval = Far_kmc.eval_field(eval_points)
    All_eval = All_kmc.eval_field(eval_points)

    err_eval_field = np.linalg.norm(Near_eval + Far_eval - All_eval, np.inf)

    assert err_eval_field < 10.**-13



def test_split_2():
    L = 6
    R = 3

    N = 200

    E = 1.
    rc = E/4

    rng = np.random.RandomState(seed=12415)

    Near = state.State()
    Near.domain = domain.BaseDomainHalo(extent=(E,E,E))
    Near.domain.boundary_condition = domain.BoundaryTypePeriodic()
    Near.npart = N
    Near.P = data.PositionDat(ncomp=3)
    Near.Q = data.ParticleDat(ncomp=1)
    Near.I = data.ParticleDat(ncomp=1, dtype=INT64)

    Far = state.State()
    Far.domain = domain.BaseDomainHalo(extent=(E,E,E))
    Far.domain.boundary_condition = domain.BoundaryTypePeriodic()
    Far.npart = N
    Far.P = data.PositionDat(ncomp=3)
    Far.Q = data.ParticleDat(ncomp=1)
    Far.I = data.ParticleDat(ncomp=1, dtype=INT64)

    All = state.State()
    All.domain = domain.BaseDomainHalo(extent=(E,E,E))
    All.domain.boundary_condition = domain.BoundaryTypePeriodic()
    All.npart = N
    All.P = data.PositionDat(ncomp=3)
    All.Q = data.ParticleDat(ncomp=1)
    All.I = data.ParticleDat(ncomp=1, dtype=INT64)

    pi = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,3))
    qi = rng.uniform(size=(N, 1))
    qi -= np.sum(qi) / N
    gi = np.arange(N).reshape((N, 1))

    with Far.modify() as mv:
        if MPIRANK == 0:
            mv.add(
                {
                    Far.P: pi,
                    Far.Q: qi,
                    Far.I: gi,
                }
            )
    with Near.modify() as mv:
        if MPIRANK == 0:
            mv.add(
                {
                    Near.P: pi,
                    Near.Q: qi,
                    Near.I: gi,
                }
            )
    with All.modify() as mv:
        if MPIRANK == 0:
            mv.add(
                {
                    All.P: pi,
                    All.Q: qi,
                    All.I: gi
                }
            )

    Near_kmc = kmc_fmm.KMCFMM(Near.P, Near.Q, Near.domain, boundary_condition='27', l=L, r=R)
    Far_kmc = kmc_fmm.KMCFMM(Far.P, Far.Q, Far.domain, boundary_condition='ff-only', l=L, r=R)
    All_kmc = kmc_fmm.KMCFMM(All.P, All.Q, All.domain, boundary_condition='pbc', l=L, r=R)

    Near_kmc.initialise()
    Far_kmc.initialise()
    All_kmc.initialise()

    Nsteps = 20
    Nprop = 4

    for stepx in range(Nsteps):
        global_proposed_moves = [rng.uniform(low=-0.5*E, high=0.5*E, size=(Nprop,3)) for px in range(N)]

        # All
        All_proposed_moves = [(lidx, global_proposed_moves[All.I[lidx, 0]] ) for lidx in range(All.npart_local)]
        All_proposed_energy = All_kmc.propose(All_proposed_moves)

        # Near 
        Near_proposed_moves = [(lidx, global_proposed_moves[Near.I[lidx, 0]] ) for lidx in range(Near.npart_local)]
        Near_proposed_energy = Near_kmc.propose(Near_proposed_moves)

        # Far
        Far_proposed_moves = [(lidx, global_proposed_moves[Far.I[lidx, 0]] ) for lidx in range(Far.npart_local)]
        Far_proposed_energy = Far_kmc.propose(Far_proposed_moves)

        correct = np.zeros((N, Nprop), REAL)
        to_test_near = np.zeros((N, Nprop), REAL)
        to_test_far = np.zeros((N, Nprop), REAL)

        for lid in range(All.npart_local):
            correct[All.I[lid, 0], :] = All_proposed_energy[lid][:]

        for lid in range(Near.npart_local):
            to_test_near[Near.I[lid, 0], :] = Near_proposed_energy[lid][:]

        for lid in range(Far.npart_local):
            to_test_far[Far.I[lid, 0], :] = Far_proposed_energy[lid][:]

        err_proposed_energy = np.linalg.norm(to_test_near + to_test_far - correct, np.inf)
        
        assert err_proposed_energy < 10.**-12

        gid_to_accept = rng.randint(N)
        mov_to_accept = rng.randint(Nprop)

        # All
        lid_loc = np.where(All.I.view[:, 0] == gid_to_accept)[0]
        if len(lid_loc) > 0:
            lid = lid_loc[0]
            All_kmc.accept((lid, global_proposed_moves[gid_to_accept][mov_to_accept, :]))
        else:
            All_kmc.accept(None)           

        # Near
        lid_loc = np.where(Near.I.view[:, 0] == gid_to_accept)[0]
        if len(lid_loc) > 0:
            lid = lid_loc[0]
            Near_kmc.accept((lid, global_proposed_moves[gid_to_accept][mov_to_accept, :]))
        else:
            Near_kmc.accept(None)

        # Far
        lid_loc = np.where(Far.I.view[:, 0] == gid_to_accept)[0]
        if len(lid_loc) > 0:
            lid = lid_loc[0]
            Far_kmc.accept((lid, global_proposed_moves[gid_to_accept][mov_to_accept, :]))
        else:
            Far_kmc.accept(None)           
        
        correct_reduce = np.zeros_like(correct)
        MPI.COMM_WORLD.Allreduce(correct, correct_reduce)
        correct_energy = correct_reduce[gid_to_accept, mov_to_accept]

        err_all_energy = abs(correct_energy - All_kmc.energy) / abs(correct_energy)
        assert err_all_energy < 10.**-15
        
        near_far_energy = Near_kmc.energy + Far_kmc.energy
        err_near_far_energy = abs(correct_energy - near_far_energy) / abs(correct_energy)
        assert err_near_far_energy < 10.**-15























