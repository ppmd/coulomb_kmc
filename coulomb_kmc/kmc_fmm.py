from __future__ import division, print_function, absolute_import
__author__ = "W.R.Saunders"

# python imports
from enum import Enum
from math import log, ceil, factorial, sqrt, cos, sin
import ctypes
from functools import lru_cache
import time

# pip package imports
import numpy as np
import scipy
from scipy.special import lpmv

# ppmd imports
from ppmd.coulomb.fmm import PyFMM
from ppmd.pairloop import StateHandler
from ppmd.mpi import MPI
from ppmd.data import ParticleDat

# coulomb_kmc imports
from coulomb_kmc import kmc_octal, kmc_local
from coulomb_kmc.common import BCType, PROFILE

REAL = ctypes.c_double
INT64 = ctypes.c_int64

@lru_cache(maxsize=10000)
def _cached_lpmv(*args, **kwargs):
    return lpmv(*args, **kwargs)


class _ENERGY(Enum):
    U_DIFF = 'u_diff'
    U0_DIRECT = 'u0_direct'
    U0_INDIRECT = 'u0_indirect'
    U1_DIRECT = 'u1_direct'
    U1_INDIRECT = 'u1_indirect'
    U01_SELF = 'u01_self'


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


    def __init__(self, positions, charges, domain, N=None, boundary_condition='pbc',
        r=None, shell_width=0.0, energy_unit=1.0,
        _debug=False, l=None, max_move=None, cuda_direct=False):

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
        
        self.cuda_direct = cuda_direct

        self.domain = domain
        self.positions = positions
        self.charges = charges
        self.energy = None
        self.group = positions.group
        self.energy_unit = energy_unit
        self.comm = self.fmm.tree.cart_comm

        self._cell_map = None
        self._cell_occ = None

        self._dsa = None
        self._dsb = None
        self._dsc = None

        self._bc = BCType(boundary_condition)
        
        self._hmatrix_py = np.zeros((2*self.fmm.L, 2*self.fmm.L))
        def Hfoo(nx, mx): return sqrt(float(factorial(nx - abs(mx)))/factorial(nx + abs(mx)))
        for nx in range(self.fmm.L):
            for mx in range(-nx, nx+1):
                self._hmatrix_py[nx, mx] = Hfoo(nx, mx)
        
        # TODO properly implement max_move
        max_move = 1.0
        self.max_move = max_move

        # class to collect required local expansions
        self.kmco = kmc_octal.LocalCellExpansions(self.fmm, self.max_move)

        # class to collect and redistribute particle data
        self.kmcl = kmc_local.LocalParticleData(self.fmm, self.max_move,
            boundary_condition=self._bc, cuda=self.cuda_direct)
        
        self._tmp_energies = {
            _ENERGY.U_DIFF      : np.zeros((1, 1), dtype=REAL),
            _ENERGY.U0_DIRECT   : np.zeros((1, 1), dtype=REAL),
            _ENERGY.U0_INDIRECT : np.zeros((1, 1), dtype=REAL),
            _ENERGY.U1_DIRECT   : np.zeros((1, 1), dtype=REAL),
            _ENERGY.U1_INDIRECT : np.zeros((1, 1), dtype=REAL),
            _ENERGY.U01_SELF    : np.zeros((1, 1), dtype=REAL)
        }


        self._wing = MPI.Win()
        self._ordering_buf = np.zeros(1, dtype=INT64)
        self._ordering_win = self._wing.Create(self._ordering_buf,
            disp_unit=self._ordering_buf[0].nbytes,
            comm=self.comm
        )


    # these should be the names of the final propose and accept methods.
    def propose(self, moves):
        """
        Propose moves by providing the local index of the particle and proposed new sites.
        Returns system energy of proposed moves.
        e.g. moves = ((0, np.array(((1, 0, 0), (0, 1, 0), (0, 0, 1)))), )
        should return (np.array((0.1, 0.2, 0.3)), )
        """
        return self.test_propose(moves, use_python=False)

    def accept(self):
        pass
    
    def _check_ordering_dats(self):
        # self._ordering_win
        if (not hasattr(self.group, '_kmc_fmm_order')) or \
                (self.group._kmc_fmm_order.dtype is not INT64) or \
                (self.group._kmc_fmm_order.ncomp != 1):
            self.group._kmc_fmm_order = ParticleDat(ncomp=1, dtype=INT64)
        
        nlocal = self.positions.npart_local
        
        sbuf = np.zeros_like(self._ordering_buf)
        sbuf[0] = nlocal
        rbuf = np.zeros_like(self._ordering_buf)

        self.comm.Barrier()
        self._ordering_win.Fence()
        self._ordering_win.Fetch_and_op(sbuf, rbuf, 0, 0)
        self._ordering_win.Fence()
        self.group._kmc_fmm_order[:nlocal:, 0] = np.arange(rbuf[0], rbuf[0] + nlocal)        

    def initialise(self):
        
        # get the current energy, also bins particles into cells
        self.energy = self.fmm(positions=self.positions, charges=self.charges)
        
        # get the local expansions into the correct places
        self.kmco.initialise()
        
        self._check_ordering_dats()
        self.kmcl.initialise(
            positions=self.positions,
            charges=self.charges,
            fmm_cells=self.group._fmm_cell,
            ids=self.group._kmc_fmm_order
        )

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
    
    def test_propose(self, moves, use_python=True):

        
        self._assert_init()
        
        # tmp testing...
        # the kmc_local has no non cuda code path yet
        if not use_python:
            du0, du1 = self.kmcl.propose(moves)
        
        num_particles = len(moves)
        max_num_moves = 0
        num_proposed = 0
        for movx in moves:
            movs = np.atleast_2d(movx[1])
            num_movs = movs.shape[0]
            max_num_moves = max(max_num_moves, num_movs)
            num_proposed += num_movs
        
        # check tmp energy arrays are large enough
        tmp_eng_stride = self._tmp_energy_check((num_particles, max_num_moves))
        
        hostt0 = time.time()
        tmp_index = 0
        # direct differences
        for movxi, movx in enumerate(moves):
            # get particle local id
            pid = movx[0]
            # get passed moves, number of moves
            movs = np.atleast_2d(movx[1])

            if not use_python:
                old_direct_energy = du0[movxi]
            else:
                old_direct_energy = self._direct_contrib_old(pid)

            for mxi, mx in enumerate(movs):
                self._tmp_energies[_ENERGY.U0_DIRECT][movxi, mxi] = old_direct_energy

                if not use_python:
                    new_direct_energy = du1[tmp_index]
                else:
                    new_direct_energy = self._direct_contrib_new(pid, mx)
                tmp_index += 1
                self._tmp_energies[_ENERGY.U1_DIRECT][movxi, mxi] = new_direct_energy
        
        #print("------")
        #print(self._tmp_energies[_ENERGY.U0_DIRECT])
        #print(self._tmp_energies[_ENERGY.U1_DIRECT])
        hostt1 = time.time()

        # indirect differences
        for movxi, movx in enumerate(moves):
            # get particle local id
            pid = movx[0]
            # get passed moves, number of moves
            movs = np.atleast_2d(movx[1])
            num_movs = movs.shape[0]
            old_indirect_energy = self._charge_indirect_energy_old(pid)
            for mxi, mx in enumerate(movs):
                self._tmp_energies[_ENERGY.U0_INDIRECT][movxi, mxi] = old_indirect_energy
                self._tmp_energies[_ENERGY.U1_INDIRECT][movxi, mxi] = self._charge_indirect_energy_new(pid, mx)
        
        # compute self interactions
        for movxi, movx in enumerate(moves):
            pid = movx[0]
            movs = np.atleast_2d(movx[1])
            num_movs = movs.shape[0]

            for mxi, mx in enumerate(movs):
                self._tmp_energies[_ENERGY.U01_SELF][movxi, mxi] = self._self_interaction(pid, mx)
        
        # compute differences
        self._tmp_energies[_ENERGY.U_DIFF] = \
              self._tmp_energies[_ENERGY.U1_DIRECT] \
            + self._tmp_energies[_ENERGY.U1_INDIRECT] \
            - self._tmp_energies[_ENERGY.U0_DIRECT] \
            - self._tmp_energies[_ENERGY.U0_INDIRECT] \
            - self._tmp_energies[_ENERGY.U01_SELF]

        prop_energy = []

        for movxi, movx in enumerate(moves):
            pid = movx[0]
            movs = np.atleast_2d(movx[1])
            num_movs = movs.shape[0]
            pid_prop_energy = np.zeros(num_movs)

            for mxi, mx in enumerate(movs):
                if np.linalg.norm(self.positions.data[pid, :] - mx) < 10.**-14:
                    pid_prop_energy[mxi] = self.energy
                else:
                    pid_prop_energy[mxi] = self.energy + self._tmp_energies[_ENERGY.U_DIFF][movxi, mxi]

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
        if self._bc is BCType.FREE_SPACE:
            e_tmp = q * q * self.energy_unit / np.linalg.norm(
                old_pos - prop_pos)
        
        # 26 nearest primary images
        elif self._bc in (BCType.NEAREST, BCType.PBC):
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
            if self._bc == BCType.PBC:
                lexp = self._really_long_range_diff(ix, prop_pos)
                # the origin is the centre of the domain hence no offsets are needed
                disp = KMCFMM.spherical(tuple(prop_pos))
                rlr = self.charges.data[ix, 0] * self.compute_phi_local(self.fmm.L, lexp, disp)[0]
                e_tmp -= rlr

        return e_tmp


    def _direct_contrib_new(self, ix, prop_pos):
        t0 = time.time()

        icx, icy, icz = self._get_cell(prop_pos)
        e_tmp = 0.0
        extent = self.domain.extent
        # print("HST: prop", prop_pos)
        q = self.charges.data[ix, 0] * self.energy_unit
        
        ncount = 0
        _tva = self._dsa
        _tvb = self._dsb
        _tvc = self._dsc
        

        for ox in KMCFMM._offsets:
            jcell = (icx + ox[0], icy + ox[1], icz + ox[2])
            image_mod = np.zeros(3)
            # print("\tHST: jcell", jcell)
            # in pbc we need to wrap the direct part
            if self._bc in (BCType.PBC, BCType.NEAREST):

                sl = 2 ** (self.fmm.R - 1)

                # position offset
                # in python -5//7 = -1
                image_mod = np.array([float(cx // sl)*ex for cx, ex in zip(jcell, extent)])

                # find the correct cell
                jcell = (jcell[0] % sl, jcell[1] % sl, jcell[2] % sl)

            if jcell in self._cell_map.keys():


                for jxi, jx in enumerate(self._cell_map[jcell]):
                    _diff = prop_pos - self.positions.data[jx, :] - image_mod
                    # print("\tHST: jpos", self.positions.data[jx, :], jcell)
                    _tva[ncount] = np.dot(_diff, _diff)
                    _tvb[ncount] = self.charges.data[jx, 0]
                    ncount += 1
         
        np.sqrt(_tva[:ncount:], out=_tvc[:ncount:])
        np.reciprocal(_tvc[:ncount:], out=_tva[:ncount:])
        e_tmp += np.dot(_tva[:ncount:], _tvb[:ncount:])
        
        self._profile_inc('py_direct_new', time.time() - t0)

        return e_tmp * q


    def _direct_contrib_old(self, ix):
        t0 = time.time()
        icx, icy, icz = self._get_fmm_cell(ix)
        e_tmp = 0.0
        extent = self.domain.extent

        q = self.charges.data[ix, 0] * self.energy_unit
        pos = self.positions.data[ix, :]
        for ox in KMCFMM._offsets:
            jcell = (icx + ox[0], icy + ox[1], icz + ox[2])
            image_mod = np.zeros(3)

            # in pbc we need to wrap the direct part
            if self._bc in (BCType.PBC, BCType.NEAREST):
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

                    # print("\t\tHST: jpos", self.positions.data[jx, :] + image_mod, jx)

                    if jx == ix:
                        _tvb[jxi] = 0.0
                        _tva[jxi] = 1.0
                        continue

                    _diff = pos - self.positions.data[jx, :] - image_mod
                    _tva[jxi] = np.dot(_diff, _diff)
                    _tvb[jxi] = self.charges.data[jx, 0]                   

                _tva = 1.0/np.sqrt(_tva)
                e_tmp += np.dot(_tva, _tvb)

        self._profile_inc('py_direct_old', time.time() - t0)

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
        cell = self._get_cell(prop_pos)
        lexp = self._get_local_expansion(cell)
        disp = self._get_cell_disp(cell, prop_pos)

        return self.charges.data[ix, 0] * self.compute_phi_local(self.fmm.L, lexp, disp)[0]


    def _charge_indirect_energy_old(self, ix):
        cell = self._get_fmm_cell(ix)
        lexp = self._get_local_expansion(cell)
        disp = self._get_cell_disp(cell, self.positions.data[ix,:])

        return self.charges.data[ix, 0] * self.compute_phi_local(self.fmm.L, lexp, disp)[0]


    def _get_local_expansion(self, cell):

        nor = list(self.kmco.global_to_local)
        lcn = [cx + lx for cx, lx in zip(reversed(cell), nor)]
        new_exp = self.kmco.local_expansions[lcn[0], lcn[1], lcn[2], :]
        
        # uses the local expansions directly in the fmm instance as opposed
        # to the new approach that gathers them directly
        test_old = False
        if test_old:
            ls = self.fmm.tree[self.fmm.R-1].local_grid_cube_size
            lo = self.fmm.tree[self.fmm.R-1].local_grid_offset
            lor = list(lo)
            lor.reverse()
            lc = [cx - lx for cx, lx in zip(cell, lor)]

            old_exp = self.fmm.tree_plain[self.fmm.R-1][lc[2], lc[1], lc[0], :]
            np.testing.assert_array_equal(old_exp, new_exp)

        return new_exp


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
            INT64(self.fmm.L),
            REAL(1.),
            KMCFMM._numpy_ptr(arr),
            KMCFMM._numpy_ptr(self.fmm._boundary_ident),
            KMCFMM._numpy_ptr(self.fmm._boundary_terms),
            KMCFMM._numpy_ptr(self.fmm._a),
            KMCFMM._numpy_ptr(self.fmm._ar),
            KMCFMM._numpy_ptr(self.fmm._ipower_mtl),
            KMCFMM._numpy_ptr(arr_out)
        )

        return arr_out

    def _tmp_energy_check(self, new_size):
        ts = self._tmp_energies[_ENERGY.U0_DIRECT].shape
        if ts[0] < new_size[0] or ts[1] < new_size[1]:
            self._tmp_energies = {
                _ENERGY.U_DIFF      : np.zeros(new_size, dtype=REAL),
                _ENERGY.U0_DIRECT   : np.zeros(new_size, dtype=REAL),
                _ENERGY.U0_INDIRECT : np.zeros(new_size, dtype=REAL),
                _ENERGY.U1_DIRECT   : np.zeros(new_size, dtype=REAL),
                _ENERGY.U1_INDIRECT : np.zeros(new_size, dtype=REAL),
                _ENERGY.U01_SELF    : np.zeros(new_size, dtype=REAL)
            }
        return self._tmp_energies[_ENERGY.U0_DIRECT].shape[1]

    def _profile_inc(self, key, inc):
        key = self.__class__.__name__ + ':' + key
        if key not in PROFILE.keys():
            PROFILE[key] = inc
        else:
            PROFILE[key] += inc





















