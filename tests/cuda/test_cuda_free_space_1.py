from __future__ import print_function, division

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"

import pytest, ctypes, math
from mpi4py import MPI
import numpy as np

np.set_printoptions(linewidth=200)

from ppmd import *
from ppmd.coulomb.fmm import *
from ppmd.coulomb.ewald_half import *
from scipy.special import sph_harm, lpmv
import time

from math import *

MPISIZE = MPI.COMM_WORLD.Get_size()
MPIRANK = MPI.COMM_WORLD.Get_rank()
MPIBARRIER = MPI.COMM_WORLD.Barrier
DEBUG = True
SHARED_MEMORY = 'omp'

from coulomb_kmc import *


_src = r"""
            #include <stdint.h>
            #include <stdio.h>
            #include <math.h>

            #define REAL double
            #define INT64 int64_t
            #define INT32 int32_t

            __device__ const INT32 OFFSETS[27] = {-111,-110,-109,-101,-100,-99,-91,-90,-89,-11,-10,-9,-1,0,1,9,10,11,89,90,91,99,100,101,109,110,111};
            __device__ const INT32 CELL_OFFSET_X = 3;
            __device__ const INT32 CELL_OFFSET_Y = 3;
            __device__ const INT32 CELL_OFFSET_Z = 3;
            __device__ const INT32 LSD_X = 10;
            __device__ const INT32 LSD_Y = 10;
            //__device__ const INT32 LSD_Z = 10;
            
            // only needed for the id which we don't
            // need to inspect for the new position


            __global__ void direct_new(
                
                const INT64 d_num_movs,
                const REAL  * d_positions,
                const REAL  * d_charges,
                const INT64 * d_ids,
                const INT64 * d_fmm_cells,
                const REAL  * d_pdata,
                const INT64 * d_cell_occ,
                const INT64 d_cell_stride,
                REAL * d_energy
        
            ) {
                
                const INT64 idx = threadIdx.x + blockIdx.x * blockDim.x;
                if (idx < d_num_movs){
                    // this performs a cast but saves a register per value
                    // should never overflow as more than 2**3a1 cells per side is unlikely
                    // the offsets are slowest to fastest (numpy)
                    const INT32 icx = d_fmm_cells[idx*3]   + CELL_OFFSET_X;
                    const INT32 icy = d_fmm_cells[idx*3+1] + CELL_OFFSET_Y;
                    const INT32 icz = d_fmm_cells[idx*3+2] + CELL_OFFSET_Z;
                    
                    const INT32 ic = icx + LSD_X * (icy + LSD_Y*icz);
                    const REAL ipx = d_positions[idx*3];
                    const REAL ipy = d_positions[idx*3+1];
                    const REAL ipz = d_positions[idx*3+2];

                    REAL energy_red = 0.0;

                    // loop over the jcells
                    for(INT32 jcx=0 ; jcx<27 ; jcx++){
                        const INT32 jc = ic + OFFSETS[jcx];

                        // compute the offset into the cell data
                        const INT32 offset = jc * ((INT32) d_cell_stride);

                        // loop over the particles in the j cell
                        for(INT32 jx=0 ; jx<d_cell_occ[jc] ; jx++){            
                            const REAL jpx = d_pdata[offset + jx*5+0];
                            const REAL jpy = d_pdata[offset + jx*5+1];
                            const REAL jpz = d_pdata[offset + jx*5+2];
                            const REAL jch = d_pdata[offset + jx*5+3];
        
                            energy_red += jch * rnorm3d(ipx - jpx, ipy - jpy, ipz - jpz);
                
                        }

                    }
                    //printf("GPU: tmps %f, %f\n", energy_red, ich);

                    energy_red *= d_charges[idx];
                    d_energy[idx] = energy_red;

                }
        
            }

            __global__ void direct_old(
                
                const INT64 d_num_movs,
                const REAL  * d_positions,
                const REAL  * d_charges,
                const INT64 * d_ids,
                const INT64 * d_fmm_cells,
                const REAL  * d_pdata,
                const INT64 * d_cell_occ,
                const INT64 d_cell_stride,
                REAL * d_energy
        
            ) {
                
                const INT64 idx = threadIdx.x + blockIdx.x * blockDim.x;
                if (idx < d_num_movs){
                    // this performs a cast but saves a register per value
                    // should never overflow as more than 2**3a1 cells per side is unlikely
                    // the offsets are slowest to fastest (numpy)
                    const INT32 icx = d_fmm_cells[idx*3]   + CELL_OFFSET_X;
                    const INT32 icy = d_fmm_cells[idx*3+1] + CELL_OFFSET_Y;
                    const INT32 icz = d_fmm_cells[idx*3+2] + CELL_OFFSET_Z;
                    
                    const INT32 ic = icx + LSD_X * (icy + LSD_Y*icz);
                    const REAL ipx = d_positions[idx*3];
                    const REAL ipy = d_positions[idx*3+1];
                    const REAL ipz = d_positions[idx*3+2];

                    REAL energy_red = 0.0;

                    // loop over the jcells
                    for(INT32 jcx=0 ; jcx<27 ; jcx++){
                        const INT32 jc = ic + OFFSETS[jcx];

                        // compute the offset into the cell data
                        const INT32 offset = jc * ((INT32) d_cell_stride);

                        // loop over the particles in the j cell
                        for(INT32 jx=0 ; jx<d_cell_occ[jc] ; jx++){            
                            const REAL jpx = d_pdata[offset + jx*5+0];
                            const REAL jpy = d_pdata[offset + jx*5+1];
                            const REAL jpz = d_pdata[offset + jx*5+2];
                            const REAL jch = d_pdata[offset + jx*5+3];
        
                            
                            const long long ll_jid =  __double_as_longlong(d_pdata[offset + jx*5+4]);
                            const int64_t jid = (int64_t) ll_jid;

                            // printf("\t\tGPU: jpos %f %f %f : jid %ld\n", jpx, jpy, jpz, jid);

                            if (jid != d_ids[idx]){
                                energy_red += jch * rnorm3d(ipx - jpx, ipy - jpy, ipz - jpz);
                            }
                
                        }

                    }
                    //printf("GPU: tmps %f, %f\n", energy_red, ich);

                    energy_red *= d_charges[idx];
                    d_energy[idx] = energy_red;

                }
        
            }

"""


def red(*input):
    try:
        from termcolor import colored
        return colored(*input, color='red')
    except Exception as e: return input
def green(*input):
    try:
        from termcolor import colored
        return colored(*input, color='green')
    except Exception as e: return input
def yellow(*input):
    try:
        from termcolor import colored
        return colored(*input, color='yellow')
    except Exception as e: return input

def red_tol(val, tol):
    if abs(val) > tol:
        return red(str(val))
    else:
        return green(str(val))


def get_fmm_cell(s, ix, R):
    cc = s._fmm_cell[ix][0]
    sl = 2 ** (R - 1)
    cx = cc % sl
    cycz = (cc - cx) // sl
    cy = cycz % sl
    cz = (cycz - cy) // sl
    return cx, cy, cz

def get_cell_disp(s, ix, R):
    sl = 2 ** (R - 1)
    csl = [s.domain.extent[0] / sl,
           s.domain.extent[1] / sl,
           s.domain.extent[2] / sl]
    
    es = [s.domain.extent[0] * -0.5,
          s.domain.extent[1] * -0.5,
          s.domain.extent[2] * -0.5]
    

    cc = get_fmm_cell(s, ix, R)

    ec = [esx + 0.5 * cx + ccx * cx for esx, cx, ccx in zip(es, csl, cc)]
    px = (s.P[ix, 0], s.P[ix, 1], s.P[ix, 2])
    
    disp = (px[0] - ec[0], px[1] - ec[1], px[2] - ec[2])
    sph = spherical(disp)
    
    return sph

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

def get_local_expansion(fmm, cell):
    ls = fmm.tree[fmm.R-1].local_grid_cube_size
    lo = fmm.tree[fmm.R-1].local_grid_offset
    lor = list(lo)
    lor.reverse()
    lc = [cx - lx for cx, lx in zip(cell, lor)]
    return fmm.tree_plain[fmm.R-1][lc[2], lc[1], lc[0], :]


def compute_phi_local(llimit, moments, disp_sph):

    phi_sph_re = 0.
    phi_sph_im = 0.
    def re_lm(l,m): return (l**2) + l + m
    def im_lm(l,m): return (l**2) + l +  m + llimit**2

    for lx in range(llimit):
        mrange = list(range(lx, -1, -1)) + list(range(1, lx+1))
        mrange2 = list(range(-1*lx, 1)) + list(range(1, lx+1))
        scipy_p = lpmv(mrange, lx, np.cos(disp_sph[1]))

        #print('lx', lx, '-------------')

        for mxi, mx in enumerate(mrange2):

            re_exp = np.cos(mx*disp_sph[2])
            im_exp = np.sin(mx*disp_sph[2])

            #print('mx', mx, im_exp)

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

def charge_indirect_energy(s, ix, fmm):
    cell = get_fmm_cell(s, ix, fmm.R)
    lexp = get_local_expansion(fmm, cell)
    disp = get_cell_disp(s, ix, fmm.R)
    return s.Q[ix,0] * compute_phi_local(fmm.L, lexp, disp)[0]


@pytest.mark.skipif('MPISIZE > 1')
def test_cuda_kmc_fmm_free_space_1():
    """
    Passes all proposed moves to kmc at once, then checks all outputs
    """


    eps = 10.**-5
    L = 12
    R = 3

    N = 200
    E = 4.
    rc = E/4

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()
    A.npart = N

    A.P = data.PositionDat(ncomp=3)
    A.Q = data.ParticleDat(ncomp=1)
    A.PP = data.ParticleDat(ncomp=3)

    A.crr = data.ScalarArray(ncomp=1)

    rng = np.random.RandomState(seed=8657)

    if N == 4:
        ra = 0.25 * E
        nra = -0.25 * E

        A.P[0,:] = ( 1.6,  1.6, 0.0)
        A.P[1,:] = (-1.500001,  1.499999, 0.0)
        A.P[2,:] = (-1.500001, -1.500001, 0.0)
        A.P[3,:] = ( 0.0,  0.0, 0.0)

        A.Q[0,0] = -1.
        A.Q[1,0] = 1.
        A.Q[2,0] = -1.
        A.Q[3,0] = 0.
    else:
        A.P[:] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,3))
        for px in range(N):
            A.Q[px,0] = (-1.0)**(px+1)
        bias = np.sum(A.Q[:N:, 0])/N
        A.Q[:, 0] -= bias

    A.scatter_data_from(0)
    
    A.PP[:] = A.P[:]

    def _direct():
        _phi_direct = 0.0
        # compute phi from image and surrounding 26 cells
        for ix in range(N):
            phi_part = 0.0
            for jx in range(ix+1, N):
                rij = np.linalg.norm(A.PP[jx,:] - A.PP[ix,:])
                _phi_direct += A.Q[ix, 0] * A.Q[jx, 0] /rij
        return _phi_direct
    
    
    # create a kmc instance
    kmc_fmm = KMCFMM(positions=A.P, charges=A.Q, 
        domain=A.domain, r=R, l=L, boundary_condition='free_space')
    kmc_fmm.initialise()
 
    # create a cuda kmc instance
    kmc_fmm_cuda = KMCFMM(positions=A.P, charges=A.Q, 
        domain=A.domain, r=R, l=L, boundary_condition='free_space', cuda_direct=True)
    kmc_fmm_cuda.initialise()
    assert kmc_fmm_cuda.kmcl.cuda_enabled is True


    # make  some random proposed moves
    order = rng.permutation(range(N))
    prop = []
    # for px in range(1):
    for px in range(N):

        propn = rng.randint(5, 10)
        # propn = 1
        prop.append(
            (
                order[px],
                rng.uniform(low=-0.5*E, high=0.5*E, size=(propn, 3))
            )
        )
    
    # get the energy of the proposed moves
    prop_energy = kmc_fmm.test_propose(moves=prop)
    prop_energy_cuda = kmc_fmm_cuda.test_propose(moves=prop)
    
    for px, pcx in zip(prop_energy, prop_energy_cuda):
        for engi, engx in enumerate(px):
            assert abs(engx - pcx[engi]) < 10.**-13

    return
    gpu_energy_new = kmc_fmm.kmcl._cuda_d['new_energy'].get()
    gpu_energy_old = kmc_fmm.kmcl._cuda_d['old_energy'].get()

    nprint = prop[0][1].shape[0]
    
    A.PP[prop[0][0], :] = prop[0][1][0,:]
    #de = _direct()
    #print(de, prop_energy[0])

    tmp_index = 0
    for pi, px in enumerate(prop):
        nprop = np.atleast_2d(px[1]).shape[0]
        for nx in range(nprop):

            assert abs(kmc_fmm._tmp_energies[test_kmc_fmm_enum.U0_DIRECT][pi, nx] - \
                    gpu_energy_old[pi]) < 10.**-12, "{} {}".format(pi, nx)           
            assert abs(kmc_fmm._tmp_energies[test_kmc_fmm_enum.U1_DIRECT][pi, nx] - \
                    gpu_energy_new[tmp_index + nx]) < 10.**-12, "{} {}".format(pi, nx)
        
        tmp_index += nprop




