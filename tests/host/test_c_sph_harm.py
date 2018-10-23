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
from itertools import product
MPISIZE = MPI.COMM_WORLD.Get_size()
MPIRANK = MPI.COMM_WORLD.Get_rank()
MPIBARRIER = MPI.COMM_WORLD.Barrier
DEBUG = True
SHARED_MEMORY = 'omp'


from coulomb_kmc import *
from coulomb_kmc.common import BCType
from coulomb_kmc.kmc_fmm_common import *

INT64 = kmc_octal.INT64
REAL = kmc_octal.REAL





def test_c_sph_harm_1():
    rng = np.random.RandomState(9476213)
    N = 20
    L = 20
    ncomp = (L**2)*2

    offsets = np.zeros(N, dtype=INT64)
    positions = np.array(rng.uniform(low=0., high=1.0, size=(N, 3)), dtype=REAL)
    charges = np.array(rng.uniform(size=N), dtype=REAL)
    moments = np.array(rng.uniform(low=0.0, high=1.0, size=ncomp), dtype=REAL)
    centres = np.array((0.5, 0.5, 0.5), dtype=REAL)
    energy = np.zeros(N, dtype=REAL)
    
    lib = kmc_octal.LocalCellExpansions._init_host_kernels(L)
    
    def cptr(arr):
        return arr.ctypes.get_as_parameter()

    lib(
        INT64(N),
        cptr(offsets),
        cptr(centres),
        cptr(positions),
        cptr(charges),
        cptr(moments),
        cptr(energy)
    )
    
    lee = LocalExpEval(L)

    for ix in range(N):
        pos = tuple(positions[ix, :] - centres)
        sph_coord = KMCFMM.spherical(pos)
        e_py = lee.compute_phi_local(moments, sph_coord)[0] * charges[ix]

        err = abs(e_py - energy[ix])
        
        assert err < 10.**-14


def test_c_sph_harm_2():
    rng = np.random.RandomState(9476213)
    N = 200
    L = 12
    ncomp = (L**2)*2

    offsets = np.zeros(N, dtype=INT64)
    positions = np.array(rng.uniform(low=0., high=1.0, size=(N, 3)), dtype=REAL)
    charges = np.array(rng.uniform(size=N), dtype=REAL)
    moments = np.array(rng.uniform(low=0.0, high=1.0, size=ncomp), dtype=REAL)
    centres = np.array((0.5, 0.5, 0.5), dtype=REAL)
    energy = np.zeros(N, dtype=REAL)
    
    lib = kmc_octal.LocalCellExpansions._init_host_kernels(L)
    
    def cptr(arr):
        return arr.ctypes.get_as_parameter()

    lib(
        INT64(N),
        cptr(offsets),
        cptr(centres),
        cptr(positions),
        cptr(charges),
        cptr(moments),
        cptr(energy)
    )
    
    lee = LocalExpEval(L)

    for ix in range(N):
        pos = tuple(positions[ix, :] - centres)
        sph_coord = KMCFMM.spherical(pos)
        e_py = lee.compute_phi_local(moments, sph_coord)[0] * charges[ix]

        err = abs(e_py - energy[ix])
        
        assert err < 10.**-14

