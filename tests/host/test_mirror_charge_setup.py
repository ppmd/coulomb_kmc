
import numpy as np
from ppmd import *
from coulomb_kmc import *
import ctypes
import pytest

halfmeps = 0.5 - 10.0**-15

direction_bools = (
    (True, False, False),
    (False, True, False),
    (False, False, True)
)

@pytest.mark.parametrize("direction", direction_bools)
def test_init_1(direction):

    rng = np.random.RandomState(seed=562321)
    
    N = 10
    E = 4.0

    s = state.State()
    s.npart = N
    
    extent = [E/2 if bx else E for bx in direction]

    s.domain = domain.BaseDomainHalo(extent=extent)
    s.domain.boundary_condition = domain.BoundaryTypePeriodic()

    s.p = data.PositionDat()
    s.q = data.ParticleDat(ncomp=1)
    s.i = data.ParticleDat(ncomp=4, dtype=ctypes.c_int)

    for dimx in range(3):
        s.p[:N:, dimx] = rng.uniform(low=-halfmeps*extent[dimx], high=halfmeps*extent[dimx], size=(N))
    s.q[:] = rng.uniform(low=-2, high=2, size=(N, 1))

    mcs = kmc_dirichlet_boundary.MirrorChargeSystem(direction, s, 'p', 'q')
    ms = mcs.mirror_state
    
    # check copy of charges
    assert np.linalg.norm(ms.q[:N:]-s.q[:N], np.inf) < 10.**-16
    # check copy and sign flip of charges
    assert np.linalg.norm(ms.q[N:2*N:] + s.q[:N], np.inf) < 10.**-16
    
    for dimx in range(3):
        if direction[dimx]:
            assert np.linalg.norm(ms.p[:N:,dimx] + ms.p[N:2*N:, dimx], np.inf) < 10.**-15
            assert len(ms.p[:N:,dimx][ms.p[:N:,dimx] < (-0.5) * E]) == 0
            assert len(ms.p[:N:,dimx][ms.p[:N:,dimx] > 0]) == 0
            assert len(ms.p[N:2*N:,dimx][ms.p[N:2*N:,dimx] > (0.5) * E]) == 0
            assert len(ms.p[N:2*N:,dimx][ms.p[N:2*N:,dimx] < 0        ]) == 0            
        else:
            assert np.linalg.norm(ms.p[:N:,dimx] - ms.p[N:2*N:, dimx], np.inf) < 10.**-15
            assert np.linalg.norm(ms.p[:N:,dimx] - s.p[:N:, dimx], np.inf) < 10.**-15

        
