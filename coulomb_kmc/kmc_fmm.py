
from ppmd.coulomb.fmm import *

class KMCFMM(object):
    def __init__(self, domain, N=None, boundary_condition='pbc',
        r=None, shell_width=0.0, force_unit=1.0, energy_unit=1.0,
        _debug=False, l=None):
        
        # horrible workaround to convert sensible boundary condition
        # parameter format to what exists for PyFMM
        _bc = {
            'pbc': False,
            'free_space': True
        }[boundary_condition]
        
        self.fmm = PyFMM(domain, N=N, free_space=_bc, r=r,
            shell_width=shell_width, cuda=False, cuda_levels=1,
            force_unit=force_unit, energy_unit=energy_unit,
            _debug=_debug, l=l, cuda_local=False)


