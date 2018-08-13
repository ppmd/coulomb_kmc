
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
        self.group = positions.group
        self.energy_unit = energy_unit

    # these should be the names of the final propose and accept methods.
    def propose(self):
        pass
    def accept(self):
        pass

    def initialise(self):
        self.energy = self.fmm(positions=self.positions, charges=self.charges)
    
    def _compute_energy(self):
        pass
    
    def test_propose(self, moves):
        """
        Propose moves by providing the local index of the particle and proposed new sites.
        Returns system energy of proposed moves.
        e.g. moves = ((0, np.array(((1, 0, 0), (0, 1, 0), (0, 0, 1)))), )
        should return ((0, np.array((0.1, 0.2, 0.3), )
        """
        
        prop_energy = []

        for movx in moves:
            # get particle local id
            pid = movx[0]
            # get passed moves, number of moves
            movs = np.atleast_2d(movx[1])
            num_movs = movs.shape[0]
            # space for output energy
            pid_prop_energy = np.zeros(num_movs)

            for mx in movs:
                print(pid, mx)

                # compute old energy u0
                u0_direct = self._direct_contrib_old(pid)
                print(u0_direct)
                # compute new energy u1
                
                # store difference

        return tuple(prop_energy)

    def _direct_contrib_old(self, ix):
        N = self.positions.npart_local
        icx, icy, icz = self._get_fmm_cell(ix)
        e_tmp = 0.0
        # horrible order N method to find directly interacting
        # charges
        for jx in range(N):
            if jx == ix:
                continue
            jcx, jcy, jcz = self._get_fmm_cell(jx)
            dcx = abs(icx - jcx)
            dcy = abs(icy - jcy)
            dcz = abs(icz - jcz)
            if (dcx < 2) and (dcy < 2) and (dcz < 2):
                print(self.positions[ix, :], self.positions[jx, :])
                # these charges interact directly
                e_tmp += self.energy_unit * self.charges[ix, 0] * self.charges[jx, 0] / np.linalg.norm(
                    self.positions[ix, :] - self.positions[jx, :])

        return e_tmp


    def _get_fmm_cell(self, ix):
        R = self.fmm.R
        cc = self.group._fmm_cell[ix][0]
        sl = 2 ** (R - 1)
        cx = cc % sl
        cycz = (cc - cx) // sl
        cy = cycz % sl
        cz = (cycz - cy) // sl
        return cx, cy, cz










