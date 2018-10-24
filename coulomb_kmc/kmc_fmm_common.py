


import ctypes
import numpy as np
from math import *
import scipy
from scipy.special import lpmv

REAL = ctypes.c_double
INT64 = ctypes.c_int64


from coulomb_kmc.common import BCType, PROFILE

class LocalExpEval(object):
    
    def __init__(self, L):
        self.L = L
        self._hmatrix_py = np.zeros((2*self.L, 2*self.L))
        def Hfoo(nx, mx): return sqrt(float(factorial(nx - abs(mx)))/factorial(nx + abs(mx)))
        for nx in range(self.L):
            for mx in range(-nx, nx+1):
                self._hmatrix_py[nx, mx] = Hfoo(nx, mx)

    def compute_phi_local(self, moments, disp_sph):
        """
        Computes the field at the podint disp_sph given by the local expansion 
        in moments
        """

        llimit = self.L
    
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

    def multipole_exp(self, sph, charge, arr):
        """
        For a charge at the point sph computes the multipole moments at the origin
        and appends them onto arr.
        """

        llimit = self.L
        def re_lm(l,m): return (l**2) + l + m
        def im_lm(l,m): return (l**2) + l +  m + llimit**2
        
        cosv = np.zeros(3 * llimit)
        sinv = np.zeros(3 * llimit)
        for mx in range(-llimit, llimit+1):
            cosv[mx] = cos(-1.0 * mx * sph[2])
            sinv[mx] = sin(-1.0 * mx * sph[2])

        for lx in range(self.L):
            scipy_p = lpmv(range(lx+1), lx, cos(sph[1]))
            radn = sph[0] ** lx
            for mx in range(-lx, lx+1):
                coeff = charge * radn * self._hmatrix_py[lx, mx] * scipy_p[abs(mx)] 
                arr[re_lm(lx, mx)] += cosv[mx] * coeff
                arr[im_lm(lx, mx)] += sinv[mx] * coeff











class FMMMPIDecomp:
    def _profile_inc(self, key, inc):
        key = self.__class__.__name__ + ':' + key
        if key not in PROFILE.keys():
            PROFILE[key] = inc
        else:
            PROFILE[key] += inc

    def _setup_propose(self, moves, direct=True):
        total_movs = 0
        for movx in moves:
            movs = np.atleast_2d(movx[1])
            num_movs = movs.shape[0]
            total_movs += num_movs
        
        num_particles = len(moves)

        self._resize_host_arrays(total_movs)
    
        tmp_index = 0
        for movi, movx in enumerate(moves):
            movs = np.atleast_2d(movx[1])
            num_movs = movs.shape[0]
            pid = movx[0]
            
            ts = tmp_index
            te = ts + num_movs
            if direct:
                self._cuda_h['new_ids'][ts:te:, 0]       = self.ids[pid, 0]
            self._cuda_h['new_charges'][ts:te:, :]   = self.charges[pid, 0]

            self._cuda_h['old_positions'][movi, :] = self.positions[pid, :]
            self._cuda_h['old_charges'][movi, :]   = self.charges[pid, 0]
            self._cuda_h['old_fmm_cells'][movi, 0] = self._gcell_to_lcell(
                self._get_fmm_cell(pid, self.fmm_cells)
            )
            if direct:
                self._cuda_h['old_ids'][movi, 0]       = self.ids[pid, 0]

            for ti, tix in enumerate(range(tmp_index, tmp_index + num_movs)):
                cell, offset = self._get_cell(movs[ti])
                self._cuda_h['new_positions'][tix, :] = movs[ti] + offset
                self._cuda_h['new_fmm_cells'][tix, 0] = \
                    self._gcell_to_lcell(cell)

            tmp_index += num_movs

        return total_movs, num_particles

    def _copy_to_device(self):
        assert self.cuda_enabled is True
        # copy the particle data to the device
        for keyx in self._cuda_h.keys():
            self._cuda_d[keyx] = gpuarray.to_gpu(self._cuda_h[keyx])

    def _resize_host_arrays(self, total_movs):
        if self._cuda_h['new_positions'].shape[0] < total_movs:
            for keyx in self._cuda_h.keys():
                ncomp = self._cuda_h[keyx].shape[1]
                dtype = self._cuda_h[keyx].dtype
                self._cuda_h[keyx] = np.zeros((total_movs, ncomp), dtype=dtype)

    def _gcell_to_lcell(self, cell):
        """
        convert a xyz global cell tuple to a linear index in the parallel data
        structure
        """

        cell = [cx + ox for cx, ox in \
            zip(cell, reversed(self.cell_data_offset))]

        return cell[0] + self.local_store_dims[2] * \
            (cell[1] + self.local_store_dims[1]*cell[2])
    
    def _global_cell_xyz(self, tcx):
        """get global cell index from xyz tuple"""
        gcs = self.global_cell_size
        return tcx[0] + gcs[0] * ( tcx[1] + gcs[1] * tcx[2] )

    def _get_fmm_cell(self, ix, cell_map, slow_to_fast=False):
        # produces xyz tuple by default
        R = self.fmm.R
        cc = cell_map[ix][0]
        sl = 2 ** (R - 1)
        cx = cc % sl
        cycz = (cc - cx) // sl
        cy = cycz % sl
        cz = (cycz - cy) // sl
        
        els = self.entry_local_size
        elo = self.entry_local_offset

        assert cz >= elo[0] and cz < elo[0] + els[0]
        assert cy >= elo[1] and cy < elo[1] + els[1]
        assert cx >= elo[2] and cx < elo[2] + els[2]

        if not slow_to_fast:
            return cx, cy, cz
        else:
            return cz, cy, cx

    def _get_cell(self, position):
        # produces xyz tuple
        extent = self.group.domain.extent
        ncps = (2**(self.fmm.R - 1))
        cell_widths = [ex / ncps for ex in extent]

        # convert to xyz
        ua = self.upper_allowed
        la = self.lower_allowed

        # compute position if origin was lower left not central
        spos = [0.5*ex + po for po, ex in zip(position, extent)]
        # if a charge is slightly out of the negative end of an axis this will
        # truncate to zero
        cell = [int(pcx / cwx) for pcx, cwx in zip(spos, cell_widths)]
        cell = tuple([ min(cx, 2**(self.fmm.R -1)) for cx in cell ])
        if self._bc is BCType.FREE_SPACE:
            # Proposed cell should never be over a periodic boundary, as there
            # are none.
            # Truncate down if too high on axis, if way too high this should
            # probably throw an error.
            return cell, np.array((0., 0., 0.), dtype=REAL)
        else:
            assert self._bc in (BCType.PBC, BCType.NEAREST)
            # we assume that in both 27 nearest and pbc a proposed move could
            # be over a periodic boundary
            # following the idea that a proposed move is always in the
            # simulation domain we need to shift
            # positions accordingly
            
            # correct for round towards zero
            rtzc = [-1 if px < 0 else 0 for px in spos]
            cell = [cx + rx for cx, rx in zip(cell, rtzc)]

            offset = [((1 if cx < lx else 0) if cx <= ux else -1) for \
                lx, cx, ux in zip(la, cell, ua)]

            # use the offsets to attempt to map into the region this rank has 
            # data over the boundary
            cell = [cx + ox * ncps for cx, ox in zip(cell, offset)]
            lc = [cx >= lx for cx, lx in zip(cell, la)]
            uc = [cx <= ux for cx, ux in zip(cell, ua)]
            if not (all(lc) and all(uc)):
                raise RuntimeError('Could not map position into sub-domain. \
                    Check all proposed positions are valid')

            return cell, np.array(
                [ox * ex for ox, ex in zip(offset, extent)], dtype=REAL)



