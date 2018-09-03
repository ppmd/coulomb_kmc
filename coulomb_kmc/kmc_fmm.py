from __future__ import division, print_function, absolute_import
__author__ = "W.R.Saunders"

# python imports
from enum import Enum
from math import log, ceil, factorial, sqrt, cos, sin
import ctypes
from functools import lru_cache

# pip package imports
import numpy as np
import scipy
from scipy.special import lpmv

# ppmd imports
from ppmd.coulomb.fmm import PyFMM

@lru_cache(maxsize=10000)
def _cached_lpmv(*args, **kwargs):
    return lpmv(*args, **kwargs)



class KMCFMM(object):

    _prof_time = 0.0

    _offsets = (
        ( -1, -1, -1),
        (  0, -1, -1),
        (  1, -1, -1),
        ( -1,  0, -1),
        (  0,  0, -1),
        (  1,  0, -1),
        ( -1,  1, -1),
        (  0,  1, -1),
        (  1,  1, -1),

        ( -1, -1,  0),
        (  0, -1,  0),
        (  1, -1,  0),
        ( -1,  0,  0),
        (  0,  0,  0),
        (  1,  0,  0),
        ( -1,  1,  0),
        (  0,  1,  0),
        (  1,  1,  0),

        ( -1, -1,  1),
        (  0, -1,  1),
        (  1, -1,  1),
        ( -1,  0,  1),
        (  0,  0,  1),
        (  1,  0,  1),
        ( -1,  1,  1),
        (  0,  1,  1),
        (  1,  1,  1),
    )

    class _BCType(Enum):
        PBC = 'pbc'
        FREE_SPACE = 'free_space'
        NEAREST = '27'

    def __init__(self, positions, charges, domain, N=None, boundary_condition='pbc',
        r=None, shell_width=0.0, energy_unit=1.0,
        _debug=False, l=None):
        
        # horrible workaround to convert sensible boundary condition
        # parameter format to what exists for PyFMM
        _bc = {
            'pbc': False,
            'free_space': True,
            '27': '27'
        }[boundary_condition]
        
        self.fmm = PyFMM(domain, N=N, free_space=_bc, r=r,
            shell_width=shell_width, cuda=False, cuda_levels=1,
            force_unit=1.0, energy_unit=energy_unit,
            _debug=_debug, l=l, cuda_local=False)
        
        self.domain = domain
        self.positions = positions
        self.charges = charges
        self.energy = None
        self.group = positions.group
        self.energy_unit = energy_unit

        self._cell_map = None
        self._cell_occ = None

        self._dsa = None
        self._dsb = None
        self._dsc = None

        self._bc = KMCFMM._BCType(boundary_condition)
        
        self._hmatrix_py = np.zeros((2*self.fmm.L, 2*self.fmm.L))
        def Hfoo(nx, mx): return sqrt(float(factorial(nx - abs(mx)))/factorial(nx + abs(mx)))
        for nx in range(self.fmm.L):
            for mx in range(-nx, nx+1):
                self._hmatrix_py[nx, mx] = Hfoo(nx, mx)

    # these should be the names of the final propose and accept methods.
    def propose(self):
        pass
    def accept(self):
        pass

    def initialise(self):
        self.energy = self.fmm(positions=self.positions, charges=self.charges)
        
        self._cell_map = {}
        cell_occ = 1

        for pid in range(self.positions.npart_total):
            cell = self._get_fmm_cell(pid)
            if cell in self._cell_map.keys():
                self._cell_map[cell].append(pid)
                cell_occ = max(cell_occ, len(self._cell_map[cell]))
            else:
                self._cell_map[cell] = [pid]

        self._dsa = np.zeros(27 * cell_occ)
        self._dsb = np.zeros(27 * cell_occ)
        self._dsc = np.zeros(27 * cell_occ)


    
    def _assert_init(self):
        if self._cell_map is None:
            raise RuntimeError('Run initialise before this call')
        if self.energy is None:
            raise RuntimeError('Run initialise before this call')
    
    def test_propose(self, moves):
        """
        Propose moves by providing the local index of the particle and proposed new sites.
        Returns system energy of proposed moves.
        e.g. moves = ((0, np.array(((1, 0, 0), (0, 1, 0), (0, 0, 1)))), )
        should return (np.array((0.1, 0.2, 0.3)), )
        """
        
        self._assert_init()

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
                if np.linalg.norm(self.positions.data[pid, :] - mx) < 10.**-14:
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
        """
        Compute the self interaction of the proposed move in the primary image with the old position
        in all other images.
        """
        old_pos = self.positions.data[ix, :]
        q = self.charges.data[ix, 0]
        ex = self.domain.extent

        # self interaction with primary image
        if self._bc is KMCFMM._BCType.FREE_SPACE:
            e_tmp = q * q * self.energy_unit / np.linalg.norm(
                old_pos - prop_pos)
        
        # 26 nearest primary images
        elif self._bc in (KMCFMM._BCType.NEAREST, KMCFMM._BCType.PBC):
            coeff = q * q * self.energy_unit
            e_tmp = 0.0
            for ox in KMCFMM._offsets:
                # image of old pos
                dox = np.array((ex[0] * ox[0], ex[1] * ox[1], ex[2] * ox[2]))
                iold_pos = old_pos + dox
                e_tmp +=  coeff / np.linalg.norm(iold_pos - prop_pos)

                # add back on the new self interaction ( this is a function of the domain extent
                # and can be precomputed up to the charge part )
                if ox != (0,0,0):
                    e_tmp -= coeff / np.linalg.norm(dox)
            
            # really long range part in the PBC case
            if self._bc == KMCFMM._BCType.PBC:
                lexp = self._really_long_range_diff(ix, prop_pos)
                # the origin is the centre of the domain hence no offsets are needed
                disp = KMCFMM.spherical(tuple(prop_pos))
                rlr = self.charges.data[ix, 0] * self.compute_phi_local(self.fmm.L, lexp, disp)[0]
                e_tmp -= rlr

        return e_tmp


    def _direct_contrib_new(self, ix, prop_pos):
        icx, icy, icz = self._get_cell(prop_pos)
        e_tmp = 0.0
        extent = self.domain.extent
        
        q = self.charges.data[ix, 0] * self.energy_unit
        
        ncount = 0
        _tva = self._dsa
        _tvb = self._dsb
        _tvc = self._dsc

        for ox in KMCFMM._offsets:
            jcell = (icx + ox[0], icy + ox[1], icz + ox[2])
            image_mod = np.zeros(3)

            # in pbc we need to wrap the direct part
            if self._bc in (KMCFMM._BCType.PBC, KMCFMM._BCType.NEAREST):

                sl = 2 ** (self.fmm.R - 1)

                # position offset
                # in python -5//7 = -1
                image_mod = np.array([float(cx // sl)*ex for cx, ex in zip(jcell, extent)])

                # find the correct cell
                jcell = (jcell[0] % sl, jcell[1] % sl, jcell[2] % sl)

            if jcell in self._cell_map.keys():


                for jxi, jx in enumerate(self._cell_map[jcell]):
                    _diff = prop_pos - self.positions.data[jx, :] - image_mod
                    _tva[ncount] = np.dot(_diff, _diff)
                    _tvb[ncount] = self.charges.data[jx, 0]
                    ncount += 1
         
        np.sqrt(_tva[:ncount:], out=_tvc[:ncount:])
        np.reciprocal(_tvc[:ncount:], out=_tva[:ncount:])
        e_tmp += np.dot(_tva[:ncount:], _tvb[:ncount:])

        return e_tmp * q


    def _direct_contrib_old(self, ix):
        icx, icy, icz = self._get_fmm_cell(ix)
        e_tmp = 0.0
        extent = self.domain.extent

        q = self.charges.data[ix, 0] * self.energy_unit
        pos = self.positions.data[ix, :]
        for ox in KMCFMM._offsets:
            jcell = (icx + ox[0], icy + ox[1], icz + ox[2])
            image_mod = np.zeros(3)

            # in pbc we need to wrap the direct part
            if self._bc in (KMCFMM._BCType.PBC, KMCFMM._BCType.NEAREST):
                sl = 2 ** (self.fmm.R - 1)
                # position offset
                # in python -5//7 = -1
                image_mod = np.array([float(cx // sl)*ex for cx, ex in zip(jcell, extent)])

                # find the correct cell
                jcell = (jcell[0] % sl, jcell[1] % sl, jcell[2] % sl)

            if jcell in self._cell_map.keys():
                _tva = np.zeros(len(self._cell_map[jcell]))
                _tvb = np.zeros(len(self._cell_map[jcell]))

                for jxi, jx in enumerate(self._cell_map[jcell]):
                    if jx == ix:
                        _tvb[jxi] = 0.0
                        _tva[jxi] = 1.0
                        continue

                    _diff = pos - self.positions.data[jx, :] - image_mod
                    _tva[jxi] = np.dot(_diff, _diff)
                    _tvb[jxi] = self.charges.data[jx, 0]                   

                _tva = 1.0/np.sqrt(_tva)
                e_tmp += np.dot(_tva, _tvb)


        return e_tmp * q


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

        return self.charges.data[ix, 0] * self.compute_phi_local(self.fmm.L, lexp, disp)[0]


    def _charge_indirect_energy_old(self, ix):
        s = self.group
        cell = self._get_fmm_cell(ix)
        lexp = self._get_local_expansion(cell)
        disp = self._get_cell_disp(cell, self.positions.data[ix,:])

        return self.charges.data[ix, 0] * self.compute_phi_local(self.fmm.L, lexp, disp)[0]


    def _get_local_expansion(self, cell):
        ls = self.fmm.tree[self.fmm.R-1].local_grid_cube_size
        lo = self.fmm.tree[self.fmm.R-1].local_grid_offset
        lor = list(lo)
        lor.reverse()
        lc = [cx - lx for cx, lx in zip(cell, lor)]
        return self.fmm.tree_plain[self.fmm.R-1][lc[2], lc[1], lc[0], :]


    def _get_cell_disp(self, cell, position):
        """
        Returns spherical coordinate of particle with local cell centre as an origin
        """
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
        """
        Converts the cartesian coordinates in xyz to spherical coordinates
        (radius, polar angle, longitude angle)
        """
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

    
    def compute_phi_local(self, llimit, moments, disp_sph):
        """
        Computes the field at the podint disp_sph given by the local expansion in
        moments
        """
        
        phi_sph_re = 0.
        phi_sph_im = 0.
        def re_lm(l,m): return (l**2) + l + m
        def im_lm(l,m): return (l**2) + l +  m + llimit**2

        cosv = np.zeros(3 * llimit)
        sinv = np.zeros(3 * llimit)
        for mx in range(-llimit, llimit+1):
            cosv[mx] = cos(mx * disp_sph[2])
            sinv[mx] = sin(mx * disp_sph[2])

        for lx in range(llimit):
            scipy_p = lpmv(range(lx+1), lx, np.cos(disp_sph[1]))
            irad = disp_sph[0] ** (lx)
            for mx in range(-lx, lx+1):

                val = self._hmatrix_py[lx, mx] * scipy_p[abs(mx)]

                scipy_real = cosv[mx] * val * irad
                scipy_imag = sinv[mx] * val * irad

                ppmd_mom_re = moments[re_lm(lx, mx)]
                ppmd_mom_im = moments[im_lm(lx, mx)]

                phi_sph_re += scipy_real*ppmd_mom_re - scipy_imag*ppmd_mom_im
                phi_sph_im += scipy_real*ppmd_mom_im + ppmd_mom_re*scipy_imag

        return phi_sph_re, phi_sph_im
    

    def _multipole_exp(self, sph, charge, arr):
        """
        For a charge at the point sph computes the multipole moments at the origin
        and appends them onto arr.
        """

        llimit = self.fmm.L
        def re_lm(l,m): return (l**2) + l + m
        def im_lm(l,m): return (l**2) + l +  m + llimit**2
        
        cosv = np.zeros(3 * llimit)
        sinv = np.zeros(3 * llimit)
        for mx in range(-llimit, llimit+1):
            cosv[mx] = cos(-1.0 * mx * sph[2])
            sinv[mx] = sin(-1.0 * mx * sph[2])

        for lx in range(self.fmm.L):
            scipy_p = lpmv(range(lx+1), lx, cos(sph[1]))
            radn = sph[0] ** lx
            for mx in range(-lx, lx+1):
                coeff = charge * radn * self._hmatrix_py[lx, mx] * scipy_p[abs(mx)] 
                arr[re_lm(lx, mx)] += cosv[mx] * coeff
                arr[im_lm(lx, mx)] += sinv[mx] * coeff


    def _multipole_diff(self, old_pos, new_pos, charge, arr):
        # output is in the numpy array arr
        # plan is to do all "really long range" corrections
        # as a matmul

        # remove the old charge
        disp = KMCFMM.spherical(tuple(old_pos))
        self._multipole_exp(disp, -1.0 * charge, arr)
        
        # add the new charge
        disp = KMCFMM.spherical(tuple(new_pos))
        self._multipole_exp(disp, charge, arr)

    @staticmethod
    def _numpy_ptr(arr):
        return arr.ctypes.data_as(ctypes.c_void_p)

    def _really_long_range_diff(self, ix, prop_pos):
        """
        Compute the correction in potential field from the "very well separated"
        images
        """
        
        l2 = self.fmm.L * self.fmm.L * 2
        arr = np.zeros(l2)
        arr_out = np.zeros(l2)
        
        self._multipole_diff(
            self.positions.data[ix, :],
            prop_pos,
            self.charges.data[ix, 0],
            arr
        )
        
        # use the really long range part of the fmm instance (extract this into a matrix)
        self.fmm._translate_mtl_lib['mtl_test_wrapper'](
            ctypes.c_int64(self.fmm.L),
            ctypes.c_double(1.),
            KMCFMM._numpy_ptr(arr),
            KMCFMM._numpy_ptr(self.fmm._boundary_ident),
            KMCFMM._numpy_ptr(self.fmm._boundary_terms),
            KMCFMM._numpy_ptr(self.fmm._a),
            KMCFMM._numpy_ptr(self.fmm._ar),
            KMCFMM._numpy_ptr(self.fmm._ipower_mtl),
            KMCFMM._numpy_ptr(arr_out)
        )

        return arr_out




























