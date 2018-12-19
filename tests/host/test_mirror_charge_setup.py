from ppmd import *
from coulomb_kmc import *
import ctypes


def test_init_1():
    
    N = 10
    E = 4.0

    s = state.State()
    s.npart = N
    s.domain = domain.BaseDomainHalo(extent=(E/2, E, E))
    s.domain.boundary_condition = domain.BoundaryTypePeriodic()

    s.p = data.PositionDat()
    s.q = data.ParticleDat(ncomp=1)
    s.i = data.ParticleDat(ncomp=4, dtype=ctypes.c_int)
    


    mcs = kmc_dirichlet_boundary.MirrorChargeSystem((True, False, False), s, 'p', 'q')



