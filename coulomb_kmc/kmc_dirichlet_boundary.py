"""
Module to handle dirichlet boundary conditions.
"""
__author__ = "W.R.Saunders"

import ppmd as md
import numpy as np

class MirrorChargeSystem:
    """
    Class to create a cubic system of mirror charges from an initial system.
    The intent is that the potential field on the boundaries of the original
    system are zero.
    """
    def __init__(self, dims_to_zero, state, position_name, charge_name):
        """
        :param dims_to_zero: xyz tuple of directions to make zero on boundary.
        :param state: initial state to tile and reflect.
        """

        assert type(state) == md.state.State
        self.state = state
        dims_to_zero = tuple(dims_to_zero)
        assert len(dims_to_zero) == 3
        assert type(dims_to_zero[0]) == bool
        assert type(dims_to_zero[1]) == bool
        assert type(dims_to_zero[2]) == bool

        self.dims_to_zero = dims_to_zero

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

        self._nzero = nzero
        
        self.mirror_state.npart = self.state.npart * (2**nzero)

        for datx in self.state.particle_dats:
            dat = getattr(state, datx)
            setattr(self.mirror_state, datx, type(dat)(ncomp=dat.ncomp, dtype=dat.dtype))

        # copy and reflect here
        assert md.mpi.MPI.COMM_WORLD.size == 1, "not implemented yet"

        # original data is translated and copied into 0:npart:
        offset = np.array([-0.5*(mx-ex) for mx, ex in zip(mirror_extent, state.domain.extent)])
        
        for datx in (position_name, charge_name):
            dat = getattr(state, datx)
            if type(dat) == md.data.PositionDat:
                for dimx in range(3):
                    getattr(self.mirror_state, datx)[:state.npart:, dimx] = \
                        getattr(state, datx)[:state.npart:, dimx] + offset[dimx]
            else:
                getattr(self.mirror_state, datx)[:state.npart:, :] = \
                    getattr(state, datx)[:state.npart:, :]
 
        # now need to reflect charges in sign ( assuming one direction is DBC )
        getattr(self.mirror_state, charge_name)[state.npart:self.mirror_state.npart:, 0] = -1.0 * \
            getattr(self.state, charge_name)[:state.npart:, 0]

        for dx in range(3):
            if dims_to_zero[dx]:
                pf = -1.0
            else:
                pf = 1.0

            getattr(self.mirror_state, position_name)[state.npart:self.mirror_state.npart:, dx] = pf * \
                getattr(self.state, position_name)[:state.npart:, dx]











