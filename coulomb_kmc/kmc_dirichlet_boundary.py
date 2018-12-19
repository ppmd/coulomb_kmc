"""
Module to handle dirichlet boundary conditions.
"""
__author__ = "W.R.Saunders"

import ppmd as md
import numpy as np
import ctypes

INT64 = ctypes.c_int64

MIRROR_ORIG = 0
MIRROR_X_REFLECT = 1
MIRROR_Y_REFLECT = 2
MIRROR_Z_REFLECT = 4

class MirrorChargeSystem:
    """
    Class to create a cubic system of mirror charges from an initial system.
    The intent is that the potential field on the boundaries of the original
    system are zero.
    """
    def __init__(self, dims_to_zero, state, position_name, charge_name, id_name):
        """
        :param dims_to_zero: xyz tuple of directions to make zero on boundary.
        :param state: initial state to tile and reflect.
        """
        
        map_name = 'mirror_map'

        assert type(state) == md.state.State
        self.state = state
        dims_to_zero = tuple(dims_to_zero)
        assert len(dims_to_zero) == 3
        assert type(dims_to_zero[0]) == bool
        assert type(dims_to_zero[1]) == bool
        assert type(dims_to_zero[2]) == bool

        self.dims_to_zero = dims_to_zero
        N = state.npart

        mirror_factors = tuple([2.0 if dx is True else 1.0 \
            for dx in dims_to_zero])

        mirror_extent = tuple([ex * fx for ex, fx in \
            zip(state.domain.extent, mirror_factors)])
        
        if abs(mirror_extent[0] - mirror_extent[1]) > 10.**-15 or \
                abs(mirror_extent[0] - mirror_extent[1]) > 10.**-15:
                    raise RuntimeError('Mirror charge system is not cubic, ' + \
                            'check input state.domain.extent.')

        self.mirror_state = md.state.State()
        self.mirror_state.domain = type(state.domain)(extent=mirror_extent)
        self.mirror_state.domain.boundary_condition = type(state.domain.boundary_condition)()

        nzero = sum(dims_to_zero)
        if nzero > 1:
            raise NotImplementedError('Dirichlet boundaries in only one direction are supported.')
        elif nzero == 0:
            raise RuntimeError('No directions found to zero field on boundary')

        self._nzero = nzero
        
        mirror_N = N * (2**nzero)
        self.mirror_state.npart = mirror_N

        for datx in (position_name, charge_name, id_name):
            dat = getattr(state, datx)
            setattr(self.mirror_state, datx, type(dat)(ncomp=dat.ncomp, dtype=dat.dtype))
        
        # hacky method to deduce the ParticleDat type (for easy CUDA compat)
        dat_type = type(getattr(state, charge_name))
        setattr(self.mirror_state, map_name, dat_type(ncomp=2, dtype=INT64))

        # copy and reflect here
        assert md.mpi.MPI.COMM_WORLD.size == 1, "not implemented yet"

        # original data is translated and copied into 0:npart:
        offset = np.array([-0.5*(mx-ex) for mx, ex in zip(mirror_extent, state.domain.extent)])
        
        # copy and shift the positions
        for dimx in range(3):
            getattr(self.mirror_state, position_name)[:N:, dimx] = \
                getattr(state, position_name)[:N:, dimx] + offset[dimx]
        
        # copy and shift the charges
        getattr(self.mirror_state, charge_name)[:N:, 0] = \
            getattr(state, charge_name)[:N:, 0]

        # now need to reflect charges in sign ( assuming one direction is DBC )
        getattr(self.mirror_state, charge_name)[N:mirror_N:, 0] = -1.0 * \
            getattr(self.state, charge_name)[:N:, 0]

        for dx in range(3):
            if dims_to_zero[dx]:
                pf = -1.0
            else:
                pf = 1.0
            getattr(self.mirror_state, position_name)[N:mirror_N:, dx] = pf * \
                getattr(self.mirror_state, position_name)[:N:, dx]

        # set the global id assuming that the existing ids are 0,..,N-1
        getattr(self.mirror_state, id_name)[:N:, 0] = \
            getattr(state, id_name)[:N:, 0]
        getattr(self.mirror_state, id_name)[N:mirror_N:, 0] = \
            getattr(state, id_name)[:N:, 0] + N
        
        if getattr(state, id_name).dtype is not INT64:
            raise RuntimeError('Only support global ids of type INT64, i.e. ctypes.c_int64')

        # set the mapping ints
        mirror_maps = getattr(self.mirror_state, map_name)
        orig_ids = getattr(state, id_name)
        
        mirror_maps[:N:, 0] = orig_ids[:N:, 0]
        mirror_maps[N:2*N:, 0] = orig_ids[:N:, 0]

        mirror_maps[:N:, 1] = MIRROR_ORIG
        if   dims_to_zero[0]: flag = MIRROR_X_REFLECT
        elif dims_to_zero[1]: flag = MIRROR_Y_REFLECT
        elif dims_to_zero[2]: flag = MIRROR_Z_REFLECT
        mirror_maps[N:2*N:, 1] = flag





