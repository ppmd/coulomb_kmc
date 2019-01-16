__author__ = "W.R.Saunders"


import ctypes
import numpy as np
from math import *

from functools import lru_cache

REAL = ctypes.c_double
INT64 = ctypes.c_int64

# cuda imports if possible
import ppmd
import ppmd.cuda
from ppmd.coulomb.sph_harm import *
from ppmd.lib.build import simple_lib_creator
from coulomb_kmc.common import spherical, cell_offsets
from ppmd.coulomb.fmm_pbc import LongRangeMTL

@lru_cache(maxsize=128)
def _A(n,m):
    return ((-1.0)**n)/sqrt(factorial(n-m)*factorial(n+m))
@lru_cache(maxsize=128)
def _h(j,k,n,m):
    if abs(k) > j: return 0.0
    if abs(m) > n: return 0.0
    if abs(m-k) > j+n: return 0.0
    icoeff = ((1.j)**(abs(k-m) - abs(k) - abs(m))).real
    return icoeff * _A(n, m) * _A(j, k) / (((-1.0) ** n) * _A(j+n, m-k))

def _re_lm(l, m): return l**2 + l + m


class FullLongRangeEnergy:
    def __init__(self, L, domain, local_exp_eval):
        # this should be a full PBC fmm instance
        self.domain = domain
        self._lee = local_exp_eval
        self.ncomp = 2*(L**2)
        self.half_ncomp = L**2

        self.lrc = LongRangeMTL(L, domain)

        self.multipole_exp = np.zeros(self.ncomp, dtype=REAL)
        self.local_dot_coeffs = np.zeros(self.ncomp, dtype=REAL)

    def initialise(self, positions, charges):
        assert self.domain.comm.size == 1, "need to MPI reduce coefficients"

        self.multipole_exp.fill(0)
        self.local_dot_coeffs.fill(0)

        for px in range(positions.npart_local):
            # multipole expansion for the whole cell
            self._lee.multipole_exp(
                spherical(tuple(positions[px,:])),
                charges[px, 0],
                self.multipole_exp
            )
            # dot product for the local expansion for the cell
            self._lee.dot_vec(
                spherical(tuple(positions[px,:])),
                charges[px, 0],
                self.local_dot_coeffs
            )
        
        L_tmp = np.zeros_like(self.local_dot_coeffs)
        self.lrc(self.multipole_exp, L_tmp)
        return 0.5 * np.dot(L_tmp, self.local_dot_coeffs)

    def propose(self, total_movs, num_particles, host_data, cuda_data, arr, use_python=True):
        es = host_data['exclusive_sum']
        old_pos = host_data['old_positions']
        new_pos = host_data['new_positions']
        old_chr = host_data['old_charges']
        
        assert es.dtype == INT64
        assert old_pos.dtype == REAL
        assert old_chr.dtype == REAL
        assert new_pos.dtype == REAL
        assert arr.dtype == REAL

        assert use_python
        
        # tmp vars

        to_remove = np.zeros(self.ncomp, dtype=REAL)
        prop_mexp = np.zeros_like(to_remove)

        to_remove_dot_vec = np.zeros_like(to_remove)
        dot_vec = np.zeros_like(to_remove)

        L_tmp = np.zeros_like(to_remove)


        # get current long range energy
        self.lrc(self.multipole_exp, L_tmp)
        old_energy = 0.5 * np.dot(L_tmp, self.local_dot_coeffs)

        for px in  range(num_particles):
            # assumed charge doesn't change
            charge = old_chr[px]
            opos = old_pos[px, :]

            nprop = es[px+1, 0] - es[px, 0]

            # remove old multipole expansion coeffs
            to_remove.fill(0)
            self._lee.multipole_exp(spherical(tuple(opos)), -charge, to_remove)
            
            # remove dot product coeffs
            to_remove_dot_vec.fill(0)
            self._lee.dot_vec(spherical(tuple(opos)), -charge, to_remove_dot_vec)


            for movxi, movx in enumerate(range(es[px, 0], es[px+1, 0])):
                prop_mexp[:] = self.multipole_exp[:].copy()
                npos = new_pos[movx, :]

                # compute the mutipole expansion of the proposed config
                self._lee.multipole_exp(spherical(tuple(npos)), charge, prop_mexp)
                # remove the old pos
                prop_mexp[:] += to_remove

                # do the same for the dot product vector
                dot_vec[:] = self.local_dot_coeffs.copy()
                dot_vec[:] += to_remove_dot_vec[:]

                # add on the proposed position
                self._lee.dot_vec(spherical(tuple(npos)), charge, dot_vec)
                
                # apply long range mtl
                L_tmp.fill(0)
                self.lrc(prop_mexp, L_tmp)
                
                # compute long range energy contribution
                new_energy = 0.5 * np.dot(L_tmp, dot_vec)

                arr[px, movxi] += old_energy - new_energy
    

    def accept(self, movedata):

        realdata = movedata[:7].view(dtype=REAL)

        old_position = realdata[0:3:]
        new_position = realdata[3:6:]
        charge       = realdata[6]

        # modify the multipole expansion for the coarest level
        self._lee.multipole_exp(spherical(tuple(old_position)), -charge, self.multipole_exp)
        self._lee.multipole_exp(spherical(tuple(new_position)),  charge, self.multipole_exp)

        # modify the dot product coefficients
        self._lee.dot_vec(spherical(tuple(old_position)), -charge, self.local_dot_coeffs)
        self._lee.dot_vec(spherical(tuple(new_position)),  charge, self.local_dot_coeffs)


    def eval_field(self, points, out):
        npoints = points.shape[0]
        lexp = np.zeros(self.ncomp, REAL)
        self.lrc(self.multipole_exp, lexp)

        for px in range(npoints):
            pointx = points[px, :]
            lr_tmp = self._lee.compute_phi_local(lexp, spherical(tuple(pointx)))[0]
            out[px] += lr_tmp



