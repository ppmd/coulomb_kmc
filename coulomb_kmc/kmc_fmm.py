"""
The KMCFMM class is the user facing interface to this implementation of the FMM-KMC algorithm. The underlying algorithm
of this implementation is described in *Fast electrostatic solvers for kinetic Monte Carlo simulations*.
"""

from __future__ import division, print_function, absolute_import
__author__ = "W.R.Saunders"

# python imports
from enum import Enum
from math import log, ceil, factorial, sqrt, cos, sin
import ctypes
from functools import lru_cache
from time import time

# pip package imports
import numpy as np
import scipy
from scipy.special import lpmv

# ppmd imports
from ppmd.coulomb.fmm import PyFMM
from ppmd.pairloop import StateHandler
from ppmd.mpi import MPI, AllocMem
from ppmd.data import ParticleDat

# coulomb_kmc imports
from coulomb_kmc import kmc_octal, kmc_local
from coulomb_kmc.common import BCType, PROFILE, cell_offsets, spherical
from coulomb_kmc.kmc_fmm_common import *
from coulomb_kmc.kmc_mpi_decomp import *
from coulomb_kmc.kmc_full_long_range import FullLongRangeEnergy
from coulomb_kmc.kmc_fmm_self_interaction import FMMSelfInteraction
from coulomb_kmc.kmc_expansion_tools import LocalExpEval

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


class _PY_KMCFMM(object):
    """
    This class contains python based test code and PoC code that is required by
    tests but is not the performant/MPI code path.
    """

    def _py_init(self):

        if self._cell_map is None:

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


    def eval_field(self, points):
        self._assert_init()

        # the data structures this uses are not updated by accept
        self.initialise()

        npoints = points.shape[0]
        out = np.zeros(npoints, dtype=REAL)

        for px in range(npoints):
            pointx = points[px, :]
            assert len(pointx) == 3

            direct_field = self._direct_contrib_new(None, pointx)
            indirect_field = self._charge_indirect_energy_new(None, pointx)
            # indirect_field = 0
            out[px] = direct_field + indirect_field

        if self._bc == BCType.PBC:
            self._lr_energy.eval_field(points, out)

        return out
    

    def _direct_contrib_new(self, ix, prop_pos):

        self._py_init()

        t0 = time()

        icx, icy, icz = self._get_cell(prop_pos)
        

        e_tmp = 0.0
        extent = self.domain.extent
        if ix is not None:
            q = self.charges.data[ix, 0]
        else:
            q = 1.0
        
        ncount = 0
        _tva = self._dsa
        _tvb = self._dsb
        _tvc = self._dsc
        

        for ox in cell_offsets:
            jcell = (icx + ox[0], icy + ox[1], icz + ox[2])
            image_mod = np.zeros(3)
            # print("\tHST: jcell", jcell)
            # in pbc we need to wrap the direct part
            if self._bc in (BCType.PBC, BCType.NEAREST):

                sl = 2 ** (self.fmm.R - 1)

                # position offset
                # in python -5//7 = -1
                image_mod = np.array([float(cx // sl)*ex for cx, ex in \
                    zip(jcell, extent)])

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
        
        self._profile_inc('py_direct_new', time() - t0)

        return e_tmp * q


    def _direct_contrib_old(self, ix):
        
        self._py_init()

        t0 = time()
        icx, icy, icz = self._get_fmm_cell(ix)
        e_tmp = 0.0
        extent = self.domain.extent

        q = self.charges.data[ix, 0]
        pos = self.positions.data[ix, :]
        for ox in cell_offsets:
            jcell = (icx + ox[0], icy + ox[1], icz + ox[2])
            image_mod = np.zeros(3)

            # in pbc we need to wrap the direct part
            if self._bc in (BCType.PBC, BCType.NEAREST):
                sl = 2 ** (self.fmm.R - 1)
                # position offset
                # in python -5//7 = -1
                image_mod = np.array([float(cx // sl)*ex for \
                    cx, ex in zip(jcell, extent)])

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

        self._profile_inc('py_direct_old', time() - t0)

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
    
    def _get_lin_cell(self, position):
        cell_tuple = self._get_cell(position)
        cc = 2**(self.fmm.R -1)

        return cell_tuple[0] + cc * (cell_tuple[1] + cc * cell_tuple[2])

    def _get_cell(self, position):

        extent = self.group.domain.extent
        cell_widths = [ex / (2**(self.fmm.R - 1)) for ex in extent]
        spos = [0.5*ex + po for po, ex in zip(position, extent)]
        
        # if a charge is slightly out of the negative end of an axis this will
        # truncate to zero
        cell = [int(pcx / cwx) for pcx, cwx in zip(spos, cell_widths)]
        # truncate down if too high on axis, if way too high this should 
        # probably throw an error
        cc = tuple([min(cx, (2**(self.fmm.R -1))-1) for cx in cell ])

        return cc


    def _charge_indirect_energy_new(self, ix, prop_pos):
        cell = self._get_cell(prop_pos)
        lexp = self._get_local_expansion(cell)
        disp = self._get_cell_disp(cell, prop_pos)

        if ix is not None:
            q = self.charges.data[ix, 0]
        else:
            q = 1.0

        return q * self._lee.compute_phi_local(lexp, disp)[0]


    def _charge_indirect_energy_old(self, ix):
        cell = self._get_fmm_cell(ix)
        lexp = self._get_local_expansion(cell)
        disp = self._get_cell_disp(cell, self.positions.data[ix,:])

        return self.charges.data[ix, 0] * \
            self._lee.compute_phi_local(lexp, disp)[0]


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
        Returns spherical coordinate of particle with local cell centre as an
        origin
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
        sph = spherical(disp)
        
        return sph


    def test_accept_reinit(self, move):
        # perform the move by setting the new position and reinitialising the instance
        self.positions[move[0], :] = move[1]
        self.initialise()   


    def test_propose(self, moves, use_python=True):

        self._assert_init()
        
        td0 = 0.0
        td1 = 0.0
        ti0 = 0.0
        ti1 = 0.0
        
        cmove_data = self.md.setup_propose(moves)

        tpd0 = time()

        num_particles = cmove_data[1]
        max_num_moves = 0
        for movx in moves:
            movs = np.atleast_2d(movx[1])
            num_movs = movs.shape[0]
            max_num_moves = max(max_num_moves, num_movs)
        
        # check tmp energy arrays are large enough
        self._tmp_energy_check((num_particles, max_num_moves))
        
        # direct differences
        for movxi, movx in enumerate(moves):
            # get particle local id
            pid = movx[0]
            # get passed moves, number of moves
            movs = np.atleast_2d(movx[1])


            old_indirect_energy = self._charge_indirect_energy_old(pid)
            old_direct_energy = self._direct_contrib_old(pid)

            for mxi, mx in enumerate(movs):
                self._tmp_energies[_ENERGY.U0_DIRECT][movxi, mxi] = \
                    old_direct_energy

                new_direct_energy = self._direct_contrib_new(pid, mx)

                self._tmp_energies[_ENERGY.U1_DIRECT][movxi, mxi] = \
                    new_direct_energy
        
                self._tmp_energies[_ENERGY.U0_INDIRECT][movxi, mxi] = \
                    old_indirect_energy

                new_indirect_energy = \
                    self._charge_indirect_energy_new(pid, mx)

                self._tmp_energies[_ENERGY.U1_INDIRECT][movxi, mxi] = \
                    new_indirect_energy

        tpd1 = time()

        self._profile_inc('py-direct-indirect', tpd1 - tpd0)


        self._si.propose(*tuple(list(cmove_data) + [self._tmp_energies[_ENERGY.U01_SELF]]))

        # long range calculation
        if self._bc == BCType.PBC:
            #print("LR DISABLED")
            self._lr_energy.propose(*tuple(list(cmove_data) + [self._tmp_energies[_ENERGY.U01_SELF]]))

        # compute differences

        if self.mirror_direction is None:
            self._tmp_energies[_ENERGY.U_DIFF] = \
                  self._tmp_energies[_ENERGY.U1_DIRECT] \
                + self._tmp_energies[_ENERGY.U1_INDIRECT] \
                - self._tmp_energies[_ENERGY.U0_DIRECT] \
                - self._tmp_energies[_ENERGY.U0_INDIRECT] \
                - self._tmp_energies[_ENERGY.U01_SELF]
        
        else:
            self._tmp_energies[_ENERGY.U_DIFF] = \
                  2.0 * self._tmp_energies[_ENERGY.U1_DIRECT] \
                + 2.0 * self._tmp_energies[_ENERGY.U1_INDIRECT] \
                - 2.0 * self._tmp_energies[_ENERGY.U0_DIRECT] \
                - 2.0 * self._tmp_energies[_ENERGY.U0_INDIRECT] \
                - self._tmp_energies[_ENERGY.U01_SELF]


        prop_energy = []

        for movxi, movx in enumerate(moves):
            pid = movx[0]
            movs = np.atleast_2d(movx[1])
            num_movs = movs.shape[0]
            pid_prop_energy = np.zeros(num_movs)

            for mxi, mx in enumerate(movs):
                if np.linalg.norm(self.positions.data[pid, :] - mx) < 10.**-14:
                    pid_prop_energy[mxi] = self.energy * self.energy_unit
                else:
                    pid_prop_energy[mxi] = (self.energy + self._tmp_energies[_ENERGY.U_DIFF][movxi, mxi]) * \
                        self.energy_unit

            prop_energy.append(pid_prop_energy)

        return tuple(prop_energy)











class KMCFMM(_PY_KMCFMM):
    """
    This class provides the methods required for the electrostatic energy component in a Kinetic Monte Carlo simulation.
    It is applicable to systems that are cubic with free space, fully periodic and (plate like) Dirichlet boundary
    conditions (work in progress).

    A KMCFMM instance should be initialised, which performs the initial FMM solve, by calling the **initialise** 
    method. Furthermore a KMCFMM instance should be freed by calling the **free** method.

    After **initialise** is called the current system energy is given by the **energy** property. Moves can be proposed
    with the **propose** or **propose_with_dats** methods. A move can be accepted by calling the **accept** method.
    When a move is accepted with the **accept** method the move is also enacted in the position PositionDat that
    the KMCFMM instance is initialised with.

    :arg PositionDat positions: PositionDat(ncomp=3, dtype=ctypes.c_double) containing charge positions.
    :arg ParticleDat charges: ParticleDat(ncomp=1, dtype=ctypes.c_int64) containing charge values.
    :arg int N: Number of charges.
    :arg str boundary_condition: Specific boundary condition as one of: 'pbc' for periodic boundary conditions, 'free_space' for the simulation cell in a vacuum or '27' for the simulation cell and the nearest 26 periodic images in a vacuum.
    :arg int r: Number of FMM levels (overrides heuristic choice using N).
    :arg float shell_width: Additional argument for FMM solver, can be left at the default of 0.0.
    :arg float energy_unit: Energy unit to use, defaults to 1.0.
    :arg bool _debug: Debugging flag for FMM solver, can be ignored.
    :arg int l: Number of expansion terms to use for FMM solve and propose/accept operations.
    :arg bool cuda_direct: Enable CUDA offloading of direct interactions (work in progress).
    :arg tuple mirror_direction: Tuple of bools, e.g. (True, False, False) to enable mirror charge handling for Dirichlet boundary conditions (work in progress).
    """

    def free(self):
        """
        Free the KMCFMM instance. Should be called before program termination. Failure to call this method may result in subsequent MPI race conditions.
        """

        self.fmm.free()
        del self.fmm

    def __init__(self, positions, charges, domain, N=None, boundary_condition='pbc',
        r=None, shell_width=0.0, energy_unit=1.0,
        _debug=False, l=None, max_move=None, cuda_direct=False, mirror_direction=None):


        # horrible workaround to convert sensible boundary condition
        # parameter format to what exists for PyFMM

        _bc = {
            'pbc': '27',        # PBC are handled separately now
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
        """Total system energy, valid after a call to **initialise**."""

        self.group = positions.group
        self.energy_unit = energy_unit
        self.comm = self.fmm.tree.cart_comm
        self.mirror_direction = mirror_direction

        if mirror_direction is not None and sum([int(dx) for dx in mirror_direction]) > 1:
            raise RuntimeError('Can only mirror in one direction.')

        self._cell_map = None
        self._cell_occ = None

        self._dsa = None
        self._dsb = None
        self._dsc = None

        self._bc = BCType(boundary_condition)
        
        self._lee = LocalExpEval(self.fmm.L)


        if max_move is not None:
            self.max_move = float(max_move)
        else:
            self.max_move = max(self.domain.extent[:])

        # class to handle the mpi decomposition and preprocessing of moves
        self.md = FMMMPIDecomp(self.fmm, self.max_move,
            boundary_condition=self._bc, cuda=self.cuda_direct)


        # self interaction handling class
        self._si = FMMSelfInteraction(self.fmm, domain, self._bc, self._lee, self.mirror_direction) 

        # long range calculation
        if self._bc == BCType.PBC:
            self._lr_energy = FullLongRangeEnergy(self.fmm.L, self.fmm.domain, self._lee, self.mirror_direction)

        # class to collect required local expansions
        self.kmco = kmc_octal.LocalCellExpansions(self.md)
        # class to collect and redistribute particle data

        #print("KMC FMM INIT BROKE")
        #self.kmcl = None
        self.kmcl = kmc_local.LocalParticleData(self.md)
        
        self._tmp_energies = {
            _ENERGY.U_DIFF      : np.zeros((1, 1), dtype=REAL),
            _ENERGY.U0_DIRECT   : np.zeros((1, 1), dtype=REAL),
            _ENERGY.U0_INDIRECT : np.zeros((1, 1), dtype=REAL),
            _ENERGY.U1_DIRECT   : np.zeros((1, 1), dtype=REAL),
            _ENERGY.U1_INDIRECT : np.zeros((1, 1), dtype=REAL),
            _ENERGY.U01_SELF    : np.zeros((1, 1), dtype=REAL)
        }

        self._ordering_buf = AllocMem((1,), dtype=INT64)


        self._ordering_win = None
        self._diff_lib = self._create_diff_lib()
    
    def __del__(self):
        del self._ordering_buf


    def _create_diff_lib(self):
        
        mf = "1.0" if self.mirror_direction is None else "2.0"

        src = r'''
        #define REAL double
        #define INT64 int64_t
        #define ENERGY_UNIT ({ENERGY_UNIT})

        extern "C" int diff_lib(
            const INT64 num_particles,
            const INT64 stride_si,
            const INT64 * RESTRICT exclusive_sum,
            const INT64 * RESTRICT rate_location,
            const REAL  * RESTRICT U0D,
            const REAL  * RESTRICT U0I,
            const REAL  * RESTRICT U1D,
            const REAL  * RESTRICT U1I,
            const REAL  * RESTRICT USI,
            REAL * RESTRICT UDIFF
        ){{
            #pragma omp parallel for schedule(dynamic)
            for(INT64 px=0 ; px<num_particles ; px++){{
                const INT64 es_start = exclusive_sum[px];
                const INT64 es_end   = exclusive_sum[px+1];
                const INT64 es_count = es_end - es_start;
                for(INT64 movx=0 ; movx<es_count ; movx++){{
                    const INT64 u1loc = es_start + movx;
                    UDIFF[rate_location[u1loc]] = (ENERGY_UNIT) * (
                        {MF} * (
                                + U1D[u1loc]
                                + U1I[u1loc] 
                                - U0D[px]
                                - U0I[px]
                               )
                        - USI[px*stride_si + movx]
                    );

                }}
            }}
            return 0;
        }}
        '''.format(
            ENERGY_UNIT=str(self.energy_unit),
            MF=mf
        )

        header = r'''
        #include <stdint.h>
        #include <stdio.h>
        '''

        return simple_lib_creator(header, src)['diff_lib']


    def propose_with_dats(self, site_max_counts, current_sites,
            prop_positions, prop_masks, prop_energy_diffs, diff=True):
        """
        Compute the energy difference that would occur if a proposed move was performed. For this interface each 
        charge exists at a site type. Each site type has a maximum number of proposed moves associated with it.
        The proposed moves are passed in a ParticleDat along with a mask. The change in system energy is returned in
        the ParticleDat prop_energy_diffs.

        A detailed explanation of this interface, including example bookkeeping algorithms, is given in the paper
        *Fast electrostatic solvers for kinetic Monte Carlo simulations*.


        :arg ScalarArray site_max_counts:    ScalarArray(dtype=c_int64)      (Input). Max moves associated with site type.
        :arg ParticleDat current_sites:      ParticleDat(dtype=c_int64)      (Input). Store which site each charge is currently on.
        :arg ParticleDat prop_positions:     ParticleDat(dtype=c_double)     (Input). Set of proposed positions for each charge.
        :arg ParticleDat prop_masks:         ParticleDat(dtype=c_int64)      (Input). Energy is computed for moves with mask > 0.
        :arg ParticleDat prop_energy_diffs:  ParticleDat(dtype=c_double)     (Output. Energy (difference) of each proposed move with mask > 0.
        :arg bool diff: default True. If diff is True then the prop_energy_diffs ParticleDat is populated with energy differences.
        """
        
        self._assert_init()

        t0 = time()
        if not diff:
            raise NotImplementedError()
        
        # we store an index computed using the mask dat for use in the rate dat, hence 
        # they need the same stride
        assert prop_energy_diffs.ncomp == prop_masks.ncomp
        assert prop_energy_diffs.dtype == REAL

        cmove_data = self.md.setup_propose_with_dats(site_max_counts, current_sites,
            prop_positions, prop_masks, prop_energy_diffs)

        num_particles = cmove_data[1]
        if num_particles == 0:
            return

        max_num_moves = np.max(site_max_counts[:])
        self._tmp_energy_check((num_particles, max_num_moves))

        du0, du1 = self.kmcl.propose(*cmove_data)
        iu0, iu1 = self.kmco.propose(*cmove_data)
        self._si.propose(*tuple(list(cmove_data) + [self._tmp_energies[_ENERGY.U01_SELF]]))

        # long range calculation
        if self._bc == BCType.PBC:
            self._lr_energy.propose(*tuple(list(cmove_data) + [self._tmp_energies[_ENERGY.U01_SELF]]))

        self._diff_lib(
            INT64(num_particles),
            INT64(self._tmp_energies[_ENERGY.U01_SELF].shape[1]),
            cmove_data[2]['exclusive_sum'].ctypes.get_as_parameter(),
            cmove_data[2]['rate_location'].ctypes.get_as_parameter(),
            du0.ctypes.get_as_parameter(),
            iu0.ctypes.get_as_parameter(),
            du1.ctypes.get_as_parameter(),
            iu1.ctypes.get_as_parameter(),
            self._tmp_energies[_ENERGY.U01_SELF].ctypes.get_as_parameter(),
            prop_energy_diffs.ctypes_data
        )

        t1 = time()
        
        self._profile_inc('propose_count', cmove_data[0])
        self._profile_inc('propose_with_dats', t1 - t0)


    # these should be the names of the final propose and accept methods.
    def propose(self, moves):
        """
        Propose moves by providing the local index of the particle and proposed new sites.
        NB. Returns system energy of proposed moves.
        
        
        :arg tuple moves: Tuple of moves where each element is a pair (id, moves_id) where id is a particle id and moves_id is a NumPy array of new positions.

        """
        self._assert_init()
        t0 = time()
        r = self._propose(moves)
        self._profile_inc('propose', time() - t0)
        return r


    def accept(self, move=None):
        """
        Accept a move passed as a tuple (int: px, np.array: new_pos), where px is the
        current particle local id, and new_pos is the new position e.g.
        (46, np.array(1.0, 2.0, 3.0))
        
        Note this will move the particle in the position dat used in the intialiser.

        If the KMC instance is defined with a mirror direction `move` should be a tuple
        of moves.
        (42, np.array(1.0, 2.0, 3.0)), (84, np.array(-1.0, 2.0, 3.0))

        :arg tuple move: move to accept
        """

        if self.mirror_direction is None:
            self._accept(move)
        else:
            assert len(move) == 2
            self._accept(move[0])
            self._accept(move[1], compute_energy=False)


    def _accept(self, move, compute_energy=True):

        t0 = time()
        
        movedata = np.zeros(10 , dtype=INT64)
        realdata = movedata[:7].view(dtype=REAL)

        if move is not None:
            old_position = self.positions[move[0], :]
            new_position = np.zeros(3, REAL)
            new_position[:] = move[1]
            charge = self.charges[move[0], 0]
            gid = self.md.ids[move[0], 0]
            old_fmm_cell = self.group._fmm_cell[move[0], 0]           
            new_fmm_cell = self._get_lin_cell(move[1])
            
            realdata[0:3:] = old_position
            realdata[3:6:] = new_position
            realdata[6]    = charge
            movedata[7]    = gid
            movedata[8]    = old_fmm_cell
            movedata[9]    = new_fmm_cell
            
            if compute_energy:
                new_energy = self.propose((move,))[0][0]
                self.energy = new_energy

        
        # with parallel MPI the move needs to be communicated here
        if self.comm.size > 1:
            move_key = np.zeros(3 + 10, INT64)
            move_recv = np.zeros_like(move_key)
            if move is not None:
                move_key[0] = 1
                move_key[1] = self.comm.rank

                # pack the new energy to pass to other ranks
                assert ctypes.sizeof(REAL) == ctypes.sizeof(INT64)
                move_key[2:3:].view(dtype=REAL)[0] = self.energy

                # add in the move data
                move_key[3:13:] = movedata.copy()

            
            # pass data to other ranks
            self.comm.Allreduce(move_key, move_recv)

            # determine that only one rank accepts a move
            if move_recv[0] > 1:
                raise RuntimeError("Multiple MPI ranks tried to accept moves.")
            elif move_recv[0] < 1: 
                raise RuntimeError("No moves to accept were passed.")

            #rank_with_move = move_recv[1]
            #self.comm.Bcast(movedata, rank_with_move)

            movedata = move_recv[3:13:].copy()
            realdata = movedata[:7].view(dtype=REAL)

            # if this rank packed the data, check the recv is correct
            if move is not None:
                err = np.linalg.norm(movedata - move_key[3:13:], np.inf)
                assert err < 10.**-16
            
            # set the new energy
            bcast_energy = move_recv[2:3:].view(REAL)[0]
            if move is not None:
                assert abs(bcast_energy - new_energy) < 10.**-15
            self.energy = bcast_energy


        # past here all ranks have movedata

        # Update the fmm cell (must be before the postion update)
        # as the position modify view could move the data to a new
        # MPI rank.
        
        if move is not None:
            self.group._fmm_cell[move[0]] = new_fmm_cell

        # update the position
        with self.positions.modify_view() as pm:
            if move is not None:
                pm[move[0], :] = move[1]



        self._profile_inc('propose_setup', time() - t0)
        
        t0 = time()
        self._si.accept(movedata)
        self._profile_inc('self_interaction_accept', time() - t0)
        
        t0 = time()
        self.kmcl.accept(movedata)
        self._profile_inc('local_accept', time() - t0)
        
        t0 = time()
        self.kmco.accept(movedata)
        self._profile_inc('octal_accept', time() - t0)
        
        if self._bc == BCType.PBC:
            t0 = time()
            self._lr_energy.accept(movedata)
            self._profile_inc('lr_energy_accept', time() - t0)






    def _check_ordering_dats(self):
        # self._ordering_win
        if (not hasattr(self.group, '_kmc_fmm_order')) or \
                (self.group._kmc_fmm_order.dtype is not INT64) or \
                (self.group._kmc_fmm_order.ncomp != 1):
            self.group._kmc_fmm_order = ParticleDat(ncomp=1, dtype=INT64)
        
        nlocal = self.positions.npart_local
        
        sbuf = np.zeros_like(self._ordering_buf.array)
        sbuf[0] = nlocal
        rbuf = np.zeros_like(self._ordering_buf.array)

        assert self._ordering_win is None
        self._ordering_win = MPI.Win.Create(self._ordering_buf.array,
            disp_unit=self._ordering_buf.array.itemsize,
            comm=self.comm
        )
        self._ordering_win.Lock(0, MPI.LOCK_SHARED)
        self._ordering_win.Fetch_and_op(sbuf, rbuf, 0, 0)
        self._ordering_win.Unlock(0)
        self.group._kmc_fmm_order[:nlocal:, 0] = np.arange(rbuf[0], rbuf[0] + nlocal)
        
        self.comm.Barrier()
        self._ordering_win.Free()
        self._ordering_win = None


    def initialise(self):
        """
        Perform the initial FMM solve and initialise the KMCFMM instance. Must be called before propose or accept can be called.

        """


        t0 = time()


        self.energy = self.fmm(positions=self.positions, charges=self.charges)
        
        self._check_ordering_dats()

        self.md.initialise(
            positions=self.positions,
            charges=self.charges,
            fmm_cells=self.group._fmm_cell,
            ids=self.group._kmc_fmm_order
        )

        self.kmcl.initialise(
            positions=self.positions,
            charges=self.charges,
            fmm_cells=self.group._fmm_cell,
            ids=self.group._kmc_fmm_order
        )

        self.kmco.initialise(
            positions=self.positions,
            charges=self.charges,
            fmm_cells=self.group._fmm_cell
        )

        self._si.initialise()

        # long range calculation
        if self._bc == BCType.PBC:
            lr_energy = self._lr_energy.initialise(positions=self.positions, charges=self.charges)
            self.energy += lr_energy

        self._profile_inc('initialise', time() - t0)

        # ensure re-init of python-test structs
        self._cell_map = None

    
    def _assert_init(self):

        if self.energy is None:
            raise RuntimeError('Run initialise before this call')



    def _propose(self, moves):

        self._assert_init()
        
        td0 = 0.0
        td1 = 0.0
        ti0 = 0.0
        ti1 = 0.0
        

        cmove_data = self.md.setup_propose(moves)

        td0 = time()
        du0, du1 = self.kmcl.propose(*cmove_data)
        td1 = time()
        
        ti0 = time()
        iu0, iu1 = self.kmco.propose(*cmove_data)
        ti1 = time()
        
        self._profile_inc('c-direct', td1 - td0)
        self._profile_inc('c-indirect', ti1 - ti0)
        self._profile_inc('propose_count', cmove_data[0])

        tpd0 = time()

        num_particles = cmove_data[1]
        max_num_moves = 0
        for movx in moves:
            movs = np.atleast_2d(movx[1])
            num_movs = movs.shape[0]
            max_num_moves = max(max_num_moves, num_movs)
        
        # check tmp energy arrays are large enough
        self._tmp_energy_check((num_particles, max_num_moves))
        
        tmp_index = 0
        for movxi, movx in enumerate(moves):

            # get passed moves, number of moves
            movs = np.atleast_2d(movx[1])
            num_movs = movs.shape[0]
            new_tmp_index = tmp_index + num_movs

            self._tmp_energies[_ENERGY.U0_DIRECT][movxi, :num_movs] = du0[movxi]
            self._tmp_energies[_ENERGY.U0_INDIRECT][movxi, :num_movs] = iu0[movxi]
            self._tmp_energies[_ENERGY.U1_DIRECT][movxi, :num_movs] = du1[tmp_index:new_tmp_index:, 0]
            self._tmp_energies[_ENERGY.U1_INDIRECT][movxi, :num_movs] = iu1[tmp_index:new_tmp_index:, 0]

            tmp_index = new_tmp_index


        tpd1 = time()
        self._profile_inc('py-direct-indirect', tpd1 - tpd0)


        self._si.propose(*tuple(list(cmove_data) + [self._tmp_energies[_ENERGY.U01_SELF]]))

        # long range calculation
        if self._bc == BCType.PBC:
            #print("LR DISABLED")
            self._lr_energy.propose(*tuple(list(cmove_data) + [self._tmp_energies[_ENERGY.U01_SELF]]))

        # compute differences

        if self.mirror_direction is None:
            self._tmp_energies[_ENERGY.U_DIFF] = \
                + self._tmp_energies[_ENERGY.U1_INDIRECT] \
                - self._tmp_energies[_ENERGY.U0_DIRECT] \
                + self._tmp_energies[_ENERGY.U1_DIRECT] \
                - self._tmp_energies[_ENERGY.U01_SELF] \
                - self._tmp_energies[_ENERGY.U0_INDIRECT]
        
        else:
            self._tmp_energies[_ENERGY.U_DIFF] = \
                  2.0 * self._tmp_energies[_ENERGY.U1_DIRECT] \
                + 2.0 * self._tmp_energies[_ENERGY.U1_INDIRECT] \
                - 2.0 * self._tmp_energies[_ENERGY.U0_DIRECT] \
                - 2.0 * self._tmp_energies[_ENERGY.U0_INDIRECT] \
                - self._tmp_energies[_ENERGY.U01_SELF]
        
        #for keyx in self._tmp_energies.keys():
        #    print(keyx, self._tmp_energies[keyx])


        prop_energy = []

        for movxi, movx in enumerate(moves):
            pid = movx[0]
            movs = np.atleast_2d(movx[1])
            num_movs = movs.shape[0]
            pid_prop_energy = np.zeros(num_movs)

            for mxi, mx in enumerate(movs):
                if np.linalg.norm(self.positions.data[pid, :] - mx) < 10.**-14:
                    pid_prop_energy[mxi] = self.energy * self.energy_unit
                else:
                    pid_prop_energy[mxi] = (self.energy + self._tmp_energies[_ENERGY.U_DIFF][movxi, mxi]) * \
                        self.energy_unit

            prop_energy.append(pid_prop_energy)

        return tuple(prop_energy)



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
















