
from ppmd.coulomb.fmm import *

class KMCFMM(object):
    def __init__(self, positions, charges, domain, N=None, boundary_condition='pbc',
        r=None, shell_width=0.0, energy_unit=1.0,
        _debug=False, l=None):
        
        # horrible workaround to convert sensible boundary condition
        # parameter format to what exists for PyFMM
        _bc = {
            'pbc': False,
            'free_space': True
        }[boundary_condition]
        
        self.fmm = PyFMM(domain, N=N, free_space=_bc, r=r,
            shell_width=shell_width, cuda=False, cuda_levels=1,
            force_unit=1.0, energy_unit=energy_unit,
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

            for mxi, mx in enumerate(movs):
                if np.linalg.norm(self.positions[pid, :] - mx) < 10.**-14:
                    pid_prop_energy[mxi] = self.energy
                else:
                    # compute old energy u0
                    u0_direct = self._direct_contrib_old(pid)
                    u0_indirect = self._charge_indirect_energy_old(pid)
                    # compute new energy u1
                    u1_direct = self._direct_contrib_new(pid, mx)
                    u1_indirect = self._charge_indirect_energy_new(pid, mx)
                    # compute any self interactions
                    u01_self = self._self_interaction(pid, mx)
                    # store difference
                    pid_prop_energy[mxi] = self.energy - u0_direct - u0_indirect + \
                        u1_direct + u1_indirect - u01_self
            
            prop_energy.append(pid_prop_energy)

        return tuple(prop_energy)
    
    
    def _self_interaction(self, ix, prop_pos):
        old_pos = self.positions[ix, :]
        q = self.charges[ix, 0]
        e_tmp = 0.0

        # put 26 self interaction code here
        
        e_tmp += q * q * self.energy_unit / np.linalg.norm(
                old_pos - prop_pos)


        return e_tmp


    def _direct_contrib_new(self, ix, prop_pos):
        N = self.positions.npart_local
        icx, icy, icz = self._get_cell(prop_pos)
        e_tmp = 0.0
        # horrible order N method to find directly interacting
        # charges
        for jx in range(N):

            jcx, jcy, jcz = self._get_fmm_cell(jx)
            dcx = abs(icx - jcx)
            dcy = abs(icy - jcy)
            dcz = abs(icz - jcz)
            if (dcx < 2) and (dcy < 2) and (dcz < 2):
                # these charges interact directly
                e_tmp += self.energy_unit * self.charges[ix, 0] * self.charges[jx, 0] / np.linalg.norm(
                    prop_pos[:] - self.positions[jx, :])
        return e_tmp


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
    
    def _get_cell(self, position):

        extent = self.group.domain.extent
        cell_widths = [ex / (2**(self.fmm.R - 1)) for ex in extent]
        spos = [0.5*ex + po for po, ex in zip(position, extent)]
        
        # if a charge is slightly out of the negative end of an axis this will
        # truncate to zero
        cell = [int(pcx / cwx) for pcx, cwx in zip(spos, cell_widths)]
        # truncate down if too high on axis, if way too high this should probably
        # throw an error
        return tuple([min(cx, 2**(self.fmm.R -1)) for cx in cell ])


    def _charge_indirect_energy_new(self, ix, prop_pos):
        s = self.group
        cell = self._get_cell(prop_pos)
        lexp = self._get_local_expansion(cell)
        disp = self._get_cell_disp(cell, prop_pos)

        return self.charges[ix, 0] * KMCFMM.compute_phi_local(self.fmm.L, lexp, disp)[0]


    def _charge_indirect_energy_old(self, ix):
        s = self.group
        cell = self._get_fmm_cell(ix)
        lexp = self._get_local_expansion(cell)
        disp = self._get_cell_disp(cell, self.positions[ix,:])

        return self.charges[ix, 0] * KMCFMM.compute_phi_local(self.fmm.L, lexp, disp)[0]


    def _get_local_expansion(self, cell):
        ls = self.fmm.tree[self.fmm.R-1].local_grid_cube_size
        lo = self.fmm.tree[self.fmm.R-1].local_grid_offset
        lor = list(lo)
        lor.reverse()
        lc = [cx - lx for cx, lx in zip(cell, lor)]
        return self.fmm.tree_plain[self.fmm.R-1][lc[2], lc[1], lc[0], :]

    def _get_cell_disp(self, cell, position):
        R = self.fmm.R
        extent = self.group.domain.extent
        sl = 2 ** (R - 1)
        csl = [extent[0] / sl,
               extent[1] / sl,
               extent[2] / sl]
        
        es = [extent[0] * -0.5,
              extent[1] * -0.5,
              extent[2] * -0.5]

        ec = [esx + 0.5 * cx + ccx * cx for esx, cx, ccx in zip(es, csl, cell)]
        
        disp = (position[0] - ec[0], position[1] - ec[1], position[2] - ec[2])
        sph = KMCFMM.spherical(disp)
        
        return sph


    @staticmethod
    def spherical(xyz):
        if type(xyz) is tuple:
            sph = np.zeros(3)
            xy = xyz[0]**2 + xyz[1]**2
            # r
            sph[0] = np.sqrt(xy + xyz[2]**2)
            # polar angle
            sph[1] = np.arctan2(np.sqrt(xy), xyz[2])
            # longitude angle
            sph[2] = np.arctan2(xyz[1], xyz[0])

        else:
            sph = np.zeros(xyz.shape)
            xy = xyz[:,0]**2 + xyz[:,1]**2
            # r
            sph[:,0] = np.sqrt(xy + xyz[:,2]**2)
            # polar angle
            sph[:,1] = np.arctan2(np.sqrt(xy), xyz[:,2])
            # longitude angle
            sph[:,2] = np.arctan2(xyz[:,1], xyz[:,0])

        return sph

    
    @staticmethod
    def compute_phi_local(llimit, moments, disp_sph):
        
        phi_sph_re = 0.
        phi_sph_im = 0.
        def re_lm(l,m): return (l**2) + l + m
        def im_lm(l,m): return (l**2) + l +  m + llimit**2

        for lx in range(llimit):
            mrange = list(range(lx, -1, -1)) + list(range(1, lx+1))
            mrange2 = list(range(-1*lx, 1)) + list(range(1, lx+1))
            scipy_p = lpmv(mrange, lx, np.cos(disp_sph[1]))

            for mxi, mx in enumerate(mrange2):

                re_exp = np.cos(mx*disp_sph[2])
                im_exp = np.sin(mx*disp_sph[2])

                val = math.sqrt(math.factorial(
                    lx - abs(mx))/math.factorial(lx + abs(mx)))
                val *= scipy_p[mxi]

                irad = disp_sph[0] ** (lx)

                scipy_real = re_exp * val * irad
                scipy_imag = im_exp * val * irad

                ppmd_mom_re = moments[re_lm(lx, mx)]
                ppmd_mom_im = moments[im_lm(lx, mx)]

                phi_sph_re += scipy_real*ppmd_mom_re - scipy_imag*ppmd_mom_im
                phi_sph_im += scipy_real*ppmd_mom_im + ppmd_mom_re*scipy_imag

        return phi_sph_re, phi_sph_im







