
from ppmd.coulomb.fmm import *

class KMCFMM(object):
    def __init__(self, positions, charges, domain, N=None, boundary_condition='pbc',
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

        self.positions = positions
        self.charges = charges
        self.energy = None

    # these should be the names of the final propose and accept methods.
    def propose():
        pass
    def accept():
        pass

    def initialise(self):
        self.energy = self.fmm(positions=self.positions, charges=self.charges)
    
    def _compute_energy(self):
        pass

    def test_propose():
        pass
