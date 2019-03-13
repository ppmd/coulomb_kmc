





import ctypes

REAL = ctypes.c_double
INT64 = ctypes.c_int64

import numpy as np


from kmc_test_common import *

from mpi4py import MPI
MPISIZE = MPI.COMM_WORLD.Get_size()
MPIRANK = MPI.COMM_WORLD.Get_rank()

def test_free_space_1():

    FSD = FreeSpaceDirect()

    for testx in range(500):

        rng = np.random.RandomState(seed=(MPIRANK+1)*93573)
        N = rng.randint(1, 100)
        
        ppi = np.zeros((N, 3), REAL)
        qi = np.zeros((N, 1), REAL)

        def _direct():
            _phi_direct = 0.0
            # compute phi from image and surrounding 26 cells
            for ix in range(N):
                for jx in range(ix+1, N):
                    rij = np.linalg.norm(ppi[jx,:] - ppi[ix,:])
                    _phi_direct += qi[ix, 0] * qi[jx, 0] / rij
            return _phi_direct


        ppi[:] = rng.uniform(-1.0, 1.0, (N,3))
        qi[:] = rng.uniform(-1.0, 1.0, (N,1))

        phi_py = _direct()
        phi_c = FSD(N, ppi, qi)

        rel = abs(phi_py)
        rel = 1.0 if rel == 0 else rel
        err = abs(phi_py - phi_c)
        assert err < 10.**-12

        


