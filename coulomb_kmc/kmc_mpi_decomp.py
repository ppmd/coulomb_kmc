import ctypes
import numpy as np
from math import *
import scipy
from scipy.special import lpmv

REAL = ctypes.c_double
INT64 = ctypes.c_int64

from coulomb_kmc.common import BCType, PROFILE
from coulomb_kmc.kmc_fmm_common import LocalOctalBase
from coulomb_kmc import kmc_octal, kmc_local

# cuda imports if possible
import ppmd
import ppmd.cuda

from ppmd.access import READ


from ppmd.mpi import MPI

from ppmd.lib.build import simple_lib_creator

if ppmd.cuda.CUDA_IMPORT:
    cudadrv = ppmd.cuda.cuda_runtime.cudadrv
    # the device should be initialised already by ppmd
    from pycuda.compiler import SourceModule
    import pycuda.gpuarray as gpuarray


class FMMMPIDecomp(LocalOctalBase):
    """
    Class that handles the MPI decomposition and converts proposed moves into
    the internal representation expected by the classes that handle the direct
    and indirect interactions.

    :arg fmm: FMM instance to use.
    :arg float max_move: Maximum distance of any hop.
    :arg boundary_condition: Boundary condition to use.
    """

    def __init__(self, fmm, max_move, boundary_condition, cuda=False):

        self.cuda_enabled = cuda

        if cuda and not ppmd.cuda.CUDA_IMPORT:
            print(ppmd.cuda.CUDA_IMPORT_ERROR)
            raise RuntimeError("CUDA was requested but failed to be initialised")

        assert boundary_condition in (BCType.PBC, BCType.FREE_SPACE, BCType.NEAREST, BCType.FF_ONLY)

        self.fmm = fmm
        self.domain = fmm.domain
        self.max_move = max_move
        self.boundary_condition = boundary_condition

        self.entry_local_size = fmm.tree.entry_map.local_size
        self.entry_local_offset = fmm.tree.entry_map.local_offset
        self.local_size = fmm.tree[-1].local_grid_cube_size
        self.local_offset = fmm.tree[-1].local_grid_offset
        self._lo_array = np.array(self.local_offset, dtype=INT64)

        self.comm = fmm.tree.cart_comm

        csc = fmm.tree.entry_map.cube_side_count
        csc = [csc, csc, csc]

        # s2f
        csw = [self.domain.extent[2] / csc[0], self.domain.extent[1] / csc[1], self.domain.extent[0] / csc[2]]

        pad = [2 + int(ceil(max_move / cx)) for cx in csw]

        ls = fmm.tree.entry_map.local_size
        lo = fmm.tree.entry_map.local_offset

        # as offset indices (of entry map)
        pad_low = [list(range(-px, 0)) for px in pad]
        pad_high = [list(range(lsx, lsx + px)) for px, lsx in zip(pad, ls)]

        # slowest to fastest to match octal tree indexing
        global_to_local = [-lo[dx] + pad[dx] for dx in range(3)]
        self.global_to_local = np.array(global_to_local, dtype=INT64)

        # cell indices as offsets (entry map not octal tree) 0 here is the first cell collected in entry map
        cell_indices = [lpx + list(range(lsx)) + hpx for lpx, lsx, hpx in zip(pad_low, ls, pad_high)]

        # cell indices as offsets from owned cells in octal tree
        entry_octal_offset = [ox - ex for ex, ox in zip(self.entry_local_offset, self.local_offset)]
        cell_indices = [[cx + ox for cx in cell_indices[dimx]] for dimx, ox in enumerate(entry_octal_offset)]

        # xyz last allowable cell offset index
        self.upper_allowed = list(reversed([cx[-2] for cx in cell_indices]))
        self.lower_allowed = list(reversed([cx[1] for cx in cell_indices]))

        self.upper_allowed_arr = np.array(self.upper_allowed, dtype=INT64)
        self.lower_allowed_arr = np.array(self.lower_allowed, dtype=INT64)

        # use cell indices to get periodic coefficients
        self.periodic_factors = [
            [(self.local_offset[di] + cellx) // csc[di] for cellx in dimx] for di, dimx in enumerate(cell_indices)
        ]

        self.cell_offsets = cell_indices

        # cell indices as actual cell indices
        cell_indices = [[(cx + osx) % cscx for cx in dx] for dx, cscx, osx in zip(cell_indices, csc, self.local_offset)]

        # this is slowest to fastest (s2f) not xyz
        local_store_dims = [len(dx) for dx in cell_indices]

        cell_data_offset = [len(px) - ox - osx for px, ox, osx in zip(pad_low, self.local_offset, entry_octal_offset)]
        self.cell_data_offset = np.array(cell_data_offset, dtype=INT64)

        # this is slowest to fastest not xyz
        self.local_store_dims = local_store_dims
        self.local_store_dims_arr = np.array(local_store_dims, dtype=INT64)

        self.global_cell_size = csc
        self.cell_indices = cell_indices

        # host copy of particle data for moves
        self._cuda_h = {}
        self._cuda_h["new_ids"] = np.zeros((1, 1), dtype=INT64)
        self._cuda_h["new_positions"] = np.zeros((1, 3), dtype=REAL)
        self._cuda_h["new_shifted_positions"] = np.zeros((1, 3), dtype=REAL)
        self._cuda_h["new_fmm_cells"] = np.zeros((1, 1), dtype=INT64)
        self._cuda_h["new_charges"] = np.zeros((1, 1), dtype=REAL)
        self._cuda_h["new_energy_d"] = np.zeros((1, 1), dtype=REAL)
        self._cuda_h["new_energy_i"] = np.zeros((1, 1), dtype=REAL)
        self._cuda_h["old_positions"] = np.zeros((1, 3), dtype=REAL)
        self._cuda_h["old_fmm_cells"] = np.zeros((1, 1), dtype=INT64)
        self._cuda_h["old_charges"] = np.zeros((1, 1), dtype=REAL)
        self._cuda_h["old_energy_d"] = np.zeros((1, 1), dtype=REAL)
        self._cuda_h["old_energy_i"] = np.zeros((1, 1), dtype=REAL)
        self._cuda_h["old_ids"] = np.zeros((1, 1), dtype=INT64)
        self._cuda_h["exclusive_sum"] = np.zeros((1, 1), dtype=INT64)
        self._cuda_h["rate_location"] = np.zeros((1, 1), dtype=INT64)

        self._cuda_d = None
        if self.cuda_enabled:
            # device copy of particle data for moves
            self._cuda_d = {}
            self._cuda_d["new_ids"] = None
            self._cuda_d["new_positions"] = None
            self._cuda_d["new_shifted_positions"] = None
            self._cuda_d["new_fmm_cells"] = None
            self._cuda_d["new_charges"] = None
            self._cuda_d["new_energy_d"] = None
            self._cuda_d["new_energy_i"] = None
            self._cuda_d["old_positions"] = None
            self._cuda_d["old_fmm_cells"] = None
            self._cuda_d["old_charges"] = None
            self._cuda_d["old_energy_d"] = None
            self._cuda_d["old_energy_i"] = None
            self._cuda_d["old_ids"] = None
            self._cuda_d["exclusive_sum"] = None
            self._cuda_d["rate_location"] = None

        self._create_dat_lib()

        self._lg2l = self.fmm.tree[-1].global_to_local
        self.win_ind = None

    def _create_win(self):
        assert self.win_ind == None
        gmap_nbytes = self.fmm.tree[-1].global_to_local.itemsize
        self.win_ind = MPI.Win.Create(self._lg2l, disp_unit=gmap_nbytes, comm=self.comm)

    def initialise(self, positions, charges, fmm_cells, ids):
        """
        Initialise the data structures for the direct interactions.

        :arg positions: Initial positions of charges.
        :arg charges: Initial charge values.
        :arg fmm_cells: FMM cells of the input charges.
        :arg ids: Unique global ids of charges.
        """

        self.positions = positions
        self.charges = charges
        self.fmm_cells = fmm_cells
        self.ids = ids
        self.group = self.positions.group

    def get_win_ind(self):
        """
        Create and return the MPI.Win for the global to local map for FMM cells.
        """
        self._create_win()
        return self.win_ind

    def free_win_ind(self):
        """
        Free the MPI.Win for the global to local map for FMM cells.
        """
        if self.win_ind is not None:
            self.win_ind.Free()
            self.win_ind = None

    def get_local_fmm_cell(self, idx):
        """
        Helper function that returns the local fmm cell linear index for a particle.

        :arg idx: Local particle index.
        """
        ct = self._get_fmm_cell(idx, self.fmm_cells)
        return self._gcell_to_lcell(ct)

    def get_local_fmm_cell_array(self, ids):
        """
        For each local id in an ctypes.c_int64 numpy array returns the linear local id in the local data 
        structure of fmm cells.

        :arg ids: NumPy array (INT64) of local ids.
        """

        assert ids.dtype == INT64
        N = ids.size

        out_buf = np.zeros_like(ids)

        ncps = 2 ** (self.fmm.R - 1)
        fmm_cells_per_side = np.array((ncps, ncps, ncps), dtype=INT64)

        self._get_fmm_lcell_lib(
            INT64(N),
            fmm_cells_per_side.ctypes.get_as_parameter(),
            self.cell_data_offset.ctypes.get_as_parameter(),
            self.local_store_dims_arr.ctypes.get_as_parameter(),
            ids.ctypes.get_as_parameter(),
            self.fmm_cells.ctypes_data_access(mode=READ, pair=False),
            out_buf.ctypes.get_as_parameter(),
        )

        return out_buf

    def setup_propose(self, moves):
        """
        Converts a tuple of proposed moves into the internal data structure for
        proposed moves.

        :arg moves: Proposed moves in tuple form. e.g. ((id, proposed_positions), ...).
        """

        total_movs = 0
        for movx in moves:
            movs = np.atleast_2d(movx[1])
            num_movs = movs.shape[0]
            total_movs += num_movs

        num_particles = len(moves)

        self._resize_host_arrays(max(total_movs, num_particles + 1))

        tmp_index = 0
        for movi, movx in enumerate(moves):
            movs = np.atleast_2d(movx[1])

            num_movs = movs.shape[0]
            pid = movx[0]

            ts = tmp_index
            te = ts + num_movs

            old_position = self.positions.data[pid, :]
            self._check_move_valid(old_position, movs)

            self._cuda_h["new_ids"][ts:te:, 0] = self.ids.data[pid, 0]
            self._cuda_h["new_charges"][ts:te:, :] = self.charges.data[pid, 0]

            self._cuda_h["old_positions"][movi, :] = old_position
            self._cuda_h["old_charges"][movi, :] = self.charges.data[pid, 0]

            try:
                ct = self._get_fmm_cell(pid, self.fmm_cells)
            except AssertionError as e:
                print("FMM cell does not appear to exist on this ranks entry map.")
                print("Local index", pid)
                print("old_position", old_position)
                print("fmm_cell", self.fmm_cells[pid, 0])
                cell = self._cell_lin_to_tuple_no_check(self.fmm_cells[pid, 0])
                print("tuple no check", cell)
                print("ELO", self.entry_local_offset)
                print("ELS", self.entry_local_size)
                raise e

            self._cuda_h["old_fmm_cells"][movi, 0] = self._gcell_to_lcell(ct)
            self._cuda_h["old_ids"][movi, 0] = self.ids.data[pid, 0]

            cells, positions, shift_pos = self._vector_get_cell(movs)

            s = tmp_index
            e = tmp_index + num_movs
            self._cuda_h["new_positions"][s:e:, :] = positions
            self._cuda_h["new_shifted_positions"][s:e:, :] = shift_pos
            self._cuda_h["new_fmm_cells"][s:e:, 0] = self._vector_gcell_to_lcell(cells)

            self._cuda_h["exclusive_sum"][movi, 0] = tmp_index
            tmp_index += num_movs

        self._cuda_h["exclusive_sum"][num_particles, 0] = tmp_index

        # for kx in self._cuda_h.keys():
        #    print(kx, self._cuda_h[kx])

        if self.cuda_enabled:
            self._copy_to_device()

        return total_movs, num_particles, self._cuda_h, self._cuda_d

    def _check_move_valid(self, old_pos, new_pos):

        e = self.domain.extent

        for dimx in range(3):
            if np.any(new_pos[:, dimx] < -0.5 * e[dimx]):
                raise RuntimeError(
                    "Proposed position is outside simulation domain. Extent {}, dimension {}, value {}.".format(
                        e, dimx, new_pos[:, dimx]
                    )
                )
            if np.any(new_pos[:, dimx] > 0.5 * e[dimx]):
                raise RuntimeError(
                    "Proposed position is outside simulation domain. Extent {}, dimension {}, value {}.".format(
                        e, dimx, new_pos[:, dimx]
                    )
                )

        old_pos = np.atleast_2d(old_pos)

        for mx in new_pos:
            rvec = mx - old_pos[0, :]
            if self.boundary_condition in (BCType.PBC, BCType.NEAREST, BCType.FF_ONLY):
                for dimx in range(3):
                    if rvec[dimx] < (e[dimx] * (-0.5)):
                        rvec[dimx] += e[dimx]
                    if rvec[dimx] > (e[dimx] * (0.5)):
                        rvec[dimx] -= e[dimx]
            rvec = np.abs(rvec)
            for dimx in range(3):
                if rvec[dimx] > self.max_move:
                    raise RuntimeError(
                        "Proposed move violates max_move. Old position {}, new position {}.".format(old_pos, mx)
                    )

    def _create_dat_lib(self):

        if self.boundary_condition is BCType.FREE_SPACE:
            cell_gen = r"""
            shifted_position[0] = position[0];
            shifted_position[1] = position[1];
            shifted_position[2] = position[2];
            """
            check_mod = r""
        else:
            assert self.boundary_condition in (BCType.PBC, BCType.NEAREST, BCType.FF_ONLY)

            cell_gen = r"""
            REAL offsets[3];

            offsets[0] = ((cell[0] - local_offset[2]) < lower_allowed[0]) ? 1.0 : 0.0;
            offsets[1] = ((cell[1] - local_offset[1]) < lower_allowed[1]) ? 1.0 : 0.0;
            offsets[2] = ((cell[2] - local_offset[0]) < lower_allowed[2]) ? 1.0 : 0.0;

            if ((cell[0] - local_offset[2]) > upper_allowed[0]) { offsets[0] = -1.0; };
            if ((cell[1] - local_offset[1]) > upper_allowed[1]) { offsets[1] = -1.0; };
            if ((cell[2] - local_offset[0]) > upper_allowed[2]) { offsets[2] = -1.0; };

            cell[0] += offsets[0] * fmm_cells_per_side[2];
            cell[1] += offsets[1] * fmm_cells_per_side[1];
            cell[2] += offsets[2] * fmm_cells_per_side[0];

            shifted_position[0] = position[0] + offsets[0] * extent[0];
            shifted_position[1] = position[1] + offsets[1] * extent[1];
            shifted_position[2] = position[2] + offsets[2] * extent[2];
            """

            check_mod = r"""
            if (d0 < (0.5*extent[0])) { d0 += extent[0]; }
            if (d1 < (0.5*extent[1])) { d1 += extent[1]; }
            if (d2 < (0.5*extent[2])) { d2 += extent[2]; }
            if (d0 > (0.5*extent[0])) { d0 -= extent[0]; }
            if (d1 > (0.5*extent[1])) { d1 -= extent[1]; }
            if (d2 > (0.5*extent[2])) { d2 -= extent[2]; }
            """

        src = r"""
        #define REAL double
        #define INT64 int64_t

        static inline void get_cell(
            const REAL * RESTRICT position,
            const REAL * RESTRICT extent,
            const INT64 * fmm_cells_per_side,
            const INT64 * RESTRICT upper_allowed,
            const INT64 * RESTRICT lower_allowed,
            const INT64 * RESTRICT local_offset,
            INT64 * cell,
            REAL * shifted_position
        ){{
            shifted_position[0] = position[0] + 0.5 * extent[0];
            shifted_position[1] = position[1] + 0.5 * extent[1];
            shifted_position[2] = position[2] + 0.5 * extent[2];

            const REAL w0 = fmm_cells_per_side[0] / extent[0];
            const REAL w1 = fmm_cells_per_side[1] / extent[1];
            const REAL w2 = fmm_cells_per_side[2] / extent[2];

            cell[0] = (INT64) (shifted_position[0] * w0);
            cell[1] = (INT64) (shifted_position[1] * w1);
            cell[2] = (INT64) (shifted_position[2] * w2);

            if (cell[0] >= fmm_cells_per_side[2]) {{ cell[0] = fmm_cells_per_side[2] - 1; }}
            if (cell[1] >= fmm_cells_per_side[1]) {{ cell[1] = fmm_cells_per_side[1] - 1; }}
            if (cell[2] >= fmm_cells_per_side[0]) {{ cell[2] = fmm_cells_per_side[0] - 1; }}

            {CELL_GEN}


            return;

        }}

        static inline void get_fmm_cell(
            const INT64 cc,
            const INT64 * fmm_cells_per_side,
            INT64 * cell
        ){{ 
            const INT64 fx = fmm_cells_per_side[0];
            const INT64 fy = fmm_cells_per_side[1];
            const INT64 cx = cc % fx;
            const INT64 cycz = (cc - cx) / fx;
            const INT64 cy = cycz % fy;
            const INT64 cz = (cycz - cy) / fy;
            cell[0] = cx;
            cell[1] = cy;
            cell[2] = cz;
            return;
        }}

        static inline INT64 gcell_to_lcell(
            const INT64 * RESTRICT cell_data_offset,
            const INT64 * RESTRICT local_store_dims,
            const INT64 * cell
        ){{
            const INT64 c0 = cell[0] + cell_data_offset[2];
            const INT64 c1 = cell[1] + cell_data_offset[1];
            const INT64 c2 = cell[2] + cell_data_offset[0];
            return c0 + local_store_dims[2] * ( c1 + local_store_dims[1] * c2 );
        }}
        
        extern "C"
        int get_fmm_lcell(
            const INT64 N,
            const INT64 * RESTRICT FMM_CELLS_PER_SIDE,
            const INT64 * RESTRICT CELL_DATA_OFFSET,
            const INT64 * RESTRICT LOCAL_STORE_DIMS,
            const INT64 * RESTRICT IDS_ARRAY,
            const INT64 * RESTRICT FMM_CELLS,
            INT64 * RESTRICT OUT_BUFF
        ){{
            
            for(INT64 ix=0 ; ix<N ; ix++){{
                INT64 tmp_cell[3];
                get_fmm_cell(FMM_CELLS[IDS_ARRAY[ix]], FMM_CELLS_PER_SIDE, tmp_cell);
                OUT_BUFF[ix] = gcell_to_lcell(CELL_DATA_OFFSET, LOCAL_STORE_DIMS, tmp_cell);
            }}
            
            return 0;
        }}
        
        static inline void check_move(
            const REAL * RESTRICT extent,
            const REAL * RESTRICT p,
            const REAL * RESTRICT pp,
            int * RESTRICT err,
            const INT64 px,
            const INT64 movx
        ){{
            
            if (pp[0] < -0.5 * extent[0]) {{ err[0]++; printf("ERROR: Proposed position is outside domain (%ld, %ld). %f %f %f -> %f %f %f\n", px, movx, p[0], p[1], p[2], pp[0], pp[1], pp[2]);}}
            if (pp[1] < -0.5 * extent[1]) {{ err[0]++; printf("ERROR: Proposed position is outside domain (%ld, %ld). %f %f %f -> %f %f %f\n", px, movx, p[0], p[1], p[2], pp[0], pp[1], pp[2]);}}
            if (pp[2] < -0.5 * extent[2]) {{ err[0]++; printf("ERROR: Proposed position is outside domain (%ld, %ld). %f %f %f -> %f %f %f\n", px, movx, p[0], p[1], p[2], pp[0], pp[1], pp[2]);}}
            if (pp[0] >  0.5 * extent[0]) {{ err[0]++; printf("ERROR: Proposed position is outside domain (%ld, %ld). %f %f %f -> %f %f %f\n", px, movx, p[0], p[1], p[2], pp[0], pp[1], pp[2]);}}
            if (pp[1] >  0.5 * extent[1]) {{ err[0]++; printf("ERROR: Proposed position is outside domain (%ld, %ld). %f %f %f -> %f %f %f\n", px, movx, p[0], p[1], p[2], pp[0], pp[1], pp[2]);}}
            if (pp[2] >  0.5 * extent[2]) {{ err[0]++; printf("ERROR: Proposed position is outside domain (%ld, %ld). %f %f %f -> %f %f %f\n", px, movx, p[0], p[1], p[2], pp[0], pp[1], pp[2]);}}
            
            REAL d0 = p[0] - pp[0];
            REAL d1 = p[1] - pp[1];
            REAL d2 = p[2] - pp[2];

            {CHECK_MOD}

            if ( (d0*d0) > ({MAX_MOVE_2}) ){{ err[0]++; printf("ERROR: Proposed move violates max_move (%ld, %ld). %f %f %f -> %f %f %f\n", px, movx, p[0], p[1], p[2], pp[0], pp[1], pp[2]);}}
            if ( (d1*d1) > ({MAX_MOVE_2}) ){{ err[0]++; printf("ERROR: Proposed move violates max_move (%ld, %ld). %f %f %f -> %f %f %f\n", px, movx, p[0], p[1], p[2], pp[0], pp[1], pp[2]);}}
            if ( (d2*d2) > ({MAX_MOVE_2}) ){{ err[0]++; printf("ERROR: Proposed move violates max_move (%ld, %ld). %f %f %f -> %f %f %f\n", px, movx, p[0], p[1], p[2], pp[0], pp[1], pp[2]);}}
        }}


        extern "C" int setup_move(
            const INT64 npart_local,
            const INT64 max_prop,
            const INT64 * RESTRICT site_max_counts,
            const INT64 * RESTRICT current_sites,
            const REAL  * RESTRICT current_positions,
            const REAL  * RESTRICT current_charges,
            const INT64 * RESTRICT current_ids,
            const INT64 * RESTRICT current_fmm_cells,
            const REAL  * RESTRICT prop_positions,
            const INT64 * RESTRICT prop_masks,
            const INT64            prop_charge_flag,
            const REAL  * RESTRICT prop_charges,
                  INT64 * RESTRICT rate_location,
                  REAL  * RESTRICT new_positions,
                  REAL  * RESTRICT new_charges,
                  INT64 * RESTRICT new_ids,
                  INT64 * RESTRICT new_fmm_cells,
                  REAL  * RESTRICT new_shifted_positions,
                  REAL  * RESTRICT old_positions,
                  REAL  * RESTRICT old_charges,
                  INT64 * RESTRICT old_ids,
                  INT64 * RESTRICT old_fmm_cells,
                  INT64 * RESTRICT exclusive_sum,
                  INT64 * RESTRICT num_particles,
                  INT64 * RESTRICT total_movs,
            const REAL  * RESTRICT extent,
            const INT64 * RESTRICT fmm_cells_per_side,
            const INT64 * RESTRICT cell_data_offset,
            const INT64 * RESTRICT local_store_dims,
            const INT64 * RESTRICT upper_allowed,
            const INT64 * RESTRICT lower_allowed,
            const INT64 * RESTRICT local_offset
        ){{
            
            int err = 0;
            INT64 es_tmp = 0;
            INT64 old_ind = 0;
            for(INT64 px=0 ; px<npart_local ; px++){{

                // Compute the exclusive sum
                INT64 es_inner = 0;
                INT64 prop_found = 0;
                INT64 max_prop_count = site_max_counts[current_sites[px]];
                for(INT64 movx=0 ; (movx<max_prop) && (prop_found<max_prop_count) ; movx++){{
                    const INT64 mask = prop_masks[px*max_prop + movx];
                    if (mask > 0){{
                        es_inner++;
                        prop_found++;
                    }}
                }}
                // if moves involve this particle we need the data
                if (es_inner > 0){{
                    exclusive_sum[old_ind] = es_tmp;
                    es_tmp += es_inner;
                    old_ids[old_ind] = px;
                    old_ind++;
                }}
            }}
            exclusive_sum[old_ind] = es_tmp;

            *num_particles = old_ind;
            *total_movs = es_tmp;

            // now move the data
            #pragma omp parallel for schedule(dynamic) reduction(+:err)
            for(INT64 oind=0 ; oind<old_ind; oind++ ){{
                const INT64 px = old_ids[oind];
                old_positions[oind*3 + 0] = current_positions[px*3 + 0];
                old_positions[oind*3 + 1] = current_positions[px*3 + 1];
                old_positions[oind*3 + 2] = current_positions[px*3 + 2];
                old_charges[oind] = current_charges[px];
                old_ids[oind] = current_ids[px];

                INT64 tmp_cell[3] = {{0,0,0}};
                get_fmm_cell(current_fmm_cells[px], fmm_cells_per_side, tmp_cell);
                old_fmm_cells[oind] = gcell_to_lcell(cell_data_offset, local_store_dims, &tmp_cell[0]);

                const INT64 prop_count = exclusive_sum[oind+1] - exclusive_sum[oind];
                const INT64 nstart = exclusive_sum[oind];
                INT64 prop_found = 0;
                for(INT64 movx=0 ; ((movx<max_prop) && (prop_found < prop_count)) ; movx++){{
                    const INT64 mask = prop_masks[px*max_prop + movx];
                    if(mask > 0){{
                        const INT64 nind = nstart + prop_found;
                        const INT64 prop_ind = px*max_prop*3 + (movx*3); 
                        new_positions[nind*3 + 0] = prop_positions[prop_ind + 0];
                        new_positions[nind*3 + 1] = prop_positions[prop_ind + 1];
                        new_positions[nind*3 + 2] = prop_positions[prop_ind + 2];

                        check_move(extent, &current_positions[px*3], &prop_positions[prop_ind], &err, px, movx);

                        INT64 tmp_cell[3] = {{0,0,0}};
                        REAL tmp_pos[3] = {{0.0, 0.0, 0.0}};
                        
                        get_cell( &prop_positions[prop_ind], extent, fmm_cells_per_side,
                            upper_allowed, lower_allowed, local_offset, &tmp_cell[0], &tmp_pos[0]);

                        new_fmm_cells[nind] = gcell_to_lcell(cell_data_offset, local_store_dims, &tmp_cell[0]);

                        new_shifted_positions[nind*3 + 0] = tmp_pos[0];
                        new_shifted_positions[nind*3 + 1] = tmp_pos[1];
                        new_shifted_positions[nind*3 + 2] = tmp_pos[2];
                        
                        const INT64 charge_prop_ind = px * max_prop + movx;
                        new_charges[nind] = (prop_charge_flag > 0) ? prop_charges[charge_prop_ind] : current_charges[px];

                        new_ids[nind] = current_ids[px];
                        rate_location[nind] = px*max_prop + movx;
                        prop_found++;
                    }}
                }}
            }}

            return err;
        }}

        """.format(
            CELL_GEN=cell_gen, CHECK_MOD=check_mod, MAX_MOVE_2=str(self.max_move * self.max_move)
        )

        header = r"""
        #include <stdint.h>
        #include <stdio.h>
        """

        _lib = simple_lib_creator(header, src)
        self._dat_lib = _lib["setup_move"]
        self._get_fmm_lcell_lib = _lib["get_fmm_lcell"]

    def setup_propose_with_dats(
        self, site_max_counts, current_sites, prop_positions, prop_masks, prop_energy_diffs, prop_charges=None
    ):
        """
        Converts proposed moves passed with the `propose_with_dats` interface
        into the internal data structure for proposed moves that can be passed
        to the classes for direct and indirect interactions.

        :arg site_max_counts:    ScalarArray, dtype=c_int64      Input
        :arg current_sites:      ParticleDat, dtype=c_int64      Input
        :arg prop_positions:     ParticleDat, dtype=c_double     Input
        :arg prop_masks:         ParticleDat, dtype=c_int64      Input
        :arg prop_energy_diffs:  ParticleDat, dtype=c_double     Output
        :arg prop_charges:       ParticleDat, dtype=c_double     Input (optional)


        Returns:
            tuple -- (`total_moves`, `num_particles`, `host_data`, `cuda_data`),
            where `total_moves` is the total number of proposed moves, 
            `num_particles` is the number of particles (charges) with proposed
            moves, `host_data` is the proposed move data on the host and
            `cuda_data` is the proposed move data on a CUDA device (if
            applicable).
            
            The `host_data` and `cuda_data` contain the following entries:

            ======================= =========================================================
            `rate_location`         Storage location of change in energy value.
            `new_positions`         Proposed positions.
            `new_charges`           Proposed charges.
            `new_ids`               Global ids of proposed moves.
            `new_fmm_cells`         New FMM cell containing proposed move.
            `new_shifted_positions` Proposed position shifted into halo region on sub-domain.
            `old_positions`         Original position.
            `old_charges`           Original charge.
            `old_ids`               Original global id.
            `old_fmm_cells`         Original FMM cell.
            `exclusive_sum`         Used as a map from old positions to new positions.
            ======================= =========================================================
        """

        assert prop_positions.dtype == REAL
        assert prop_masks.dtype == INT64
        assert prop_energy_diffs.dtype == REAL
        mov_stride = prop_masks.ncomp
        self._resize_host_arrays((self.positions.npart_local + 1) * mov_stride)

        assert self.positions.dtype == REAL
        assert self.charges.dtype == REAL
        assert self.ids.dtype == INT64
        assert site_max_counts.dtype == INT64
        assert current_sites.dtype == INT64

        extent = self.group.domain.extent
        assert extent.dtype == REAL

        ncps = 2 ** (self.fmm.R - 1)
        fmm_cells_per_side = np.array((ncps, ncps, ncps), dtype=INT64)

        prop_charge_flag = INT64(1 if prop_charges is not None else 0)
        prop_charges_ptr = prop_charges.ctypes_data if prop_charges is not None else ctypes.byref(REAL(0))

        total_movs = INT64(0)
        num_particles = INT64(0)
        err = self._dat_lib(
            INT64(self.positions.npart_local),
            INT64(prop_masks.ncomp),
            site_max_counts.ctypes_data,
            current_sites.ctypes_data,
            self.positions.ctypes_data,
            self.charges.ctypes_data,
            self.ids.ctypes_data,
            self.fmm_cells.ctypes_data,
            prop_positions.ctypes_data,
            prop_masks.ctypes_data,
            prop_charge_flag,
            prop_charges_ptr,
            self._cuda_h["rate_location"].ctypes.get_as_parameter(),
            self._cuda_h["new_positions"].ctypes.get_as_parameter(),
            self._cuda_h["new_charges"].ctypes.get_as_parameter(),
            self._cuda_h["new_ids"].ctypes.get_as_parameter(),
            self._cuda_h["new_fmm_cells"].ctypes.get_as_parameter(),
            self._cuda_h["new_shifted_positions"].ctypes.get_as_parameter(),
            self._cuda_h["old_positions"].ctypes.get_as_parameter(),
            self._cuda_h["old_charges"].ctypes.get_as_parameter(),
            self._cuda_h["old_ids"].ctypes.get_as_parameter(),
            self._cuda_h["old_fmm_cells"].ctypes.get_as_parameter(),
            self._cuda_h["exclusive_sum"].ctypes.get_as_parameter(),
            ctypes.byref(num_particles),
            ctypes.byref(total_movs),
            extent.ctypes_data,
            fmm_cells_per_side.ctypes.get_as_parameter(),
            self.cell_data_offset.ctypes.get_as_parameter(),
            self.local_store_dims_arr.ctypes.get_as_parameter(),
            self.upper_allowed_arr.ctypes.get_as_parameter(),
            self.lower_allowed_arr.ctypes.get_as_parameter(),
            self._lo_array.ctypes.get_as_parameter(),
        )

        if err > 0:
            raise RuntimeError("Bad proposed move was detected.")

        if self.cuda_enabled:
            self._copy_to_device()

        return total_movs.value, num_particles.value, self._cuda_h, self._cuda_d

    def _copy_to_device(self):
        assert self.cuda_enabled is True
        # copy the particle data to the device
        for keyx in self._cuda_h.keys():
            self._cuda_d[keyx] = gpuarray.to_gpu(self._cuda_h[keyx])

    def _resize_host_arrays(self, total_movs):
        if self._cuda_h["new_positions"].shape[0] < total_movs:
            for keyx in self._cuda_h.keys():
                ncomp = self._cuda_h[keyx].shape[1]
                dtype = self._cuda_h[keyx].dtype
                self._cuda_h[keyx] = np.zeros((total_movs, ncomp), dtype=dtype)

    def _gcell_to_lcell(self, cell_orig):
        """
        convert a xyz global cell tuple to a linear index in the parallel data
        structure
        """

        # cell = [cx + ox for cx, ox in \
        #     zip(cell_orig, reversed(self.cell_data_offset))]

        # tmp = cell[0] + self.local_store_dims[2] * \
        #     (cell[1] + self.local_store_dims[1]*cell[2])

        # one line version of the above
        c = (
            (cell_orig[0] + self.cell_data_offset[2])
            + ((cell_orig[1] + self.cell_data_offset[1]) * self.local_store_dims[2])
            + ((cell_orig[2] + self.cell_data_offset[0]) * self.local_store_dims[1] * self.local_store_dims[2])
        )
        return c

    def _vector_gcell_to_lcell(self, cells):
        cells[:, 0] += self.cell_data_offset[2]
        cells[:, 1] += self.cell_data_offset[1]
        cells[:, 2] += self.cell_data_offset[0]

        return cells[:, 0] + self.local_store_dims[2] * (cells[:, 1] + self.local_store_dims[1] * cells[:, 2])

    def _vector_get_cell(self, positions):
        # produces xyz tuple
        extent = self.group.domain.extent
        ncps = 2 ** (self.fmm.R - 1)
        cell_widths = [extent[0] / ncps, extent[1] / ncps, extent[2] / ncps]
        # convert to xyz
        ua = self.upper_allowed
        la = self.lower_allowed

        shift_pos = positions.copy()
        shift_pos[:, 0] += 0.5 * extent[0]
        shift_pos[:, 1] += 0.5 * extent[1]
        shift_pos[:, 2] += 0.5 * extent[2]

        cell_bin = np.zeros_like(shift_pos)
        cell_bin[:, 0] = shift_pos[:, 0] / cell_widths[0]
        cell_bin[:, 1] = shift_pos[:, 1] / cell_widths[1]
        cell_bin[:, 2] = shift_pos[:, 2] / cell_widths[2]

        cells = np.zeros(cell_bin.shape, dtype=INT64)
        cells[:] = cell_bin[:]

        # if a charge is sat right on the upper boundary we prevent it being binned into a cell
        # that is outside the domain.
        cells[:] = np.clip(cells, 0, ncps - 1)

        if self.boundary_condition is BCType.FREE_SPACE:
            return cells, positions, positions
        else:
            assert self.boundary_condition in (BCType.PBC, BCType.NEAREST, BCType.FF_ONLY)

            offset_cells = cells.copy()
            offset_cells[:, 0] -= self.local_offset[2]
            offset_cells[:, 1] -= self.local_offset[1]
            offset_cells[:, 2] -= self.local_offset[0]

            offsets = np.zeros_like(cells)
            offsets[offset_cells[:, 0] < la[0], 0] = 1
            offsets[offset_cells[:, 0] > ua[0], 0] = -1
            offsets[offset_cells[:, 1] < la[1], 1] = 1
            offsets[offset_cells[:, 1] > ua[1], 1] = -1
            offsets[offset_cells[:, 2] < la[2], 2] = 1
            offsets[offset_cells[:, 2] > ua[2], 2] = -1

            # map the position
            shift_pos[:, 0] = positions[:, 0] + offsets[:, 0] * extent[0]
            shift_pos[:, 1] = positions[:, 1] + offsets[:, 1] * extent[1]
            shift_pos[:, 2] = positions[:, 2] + offsets[:, 2] * extent[2]

            # map the fmm cell
            for dx in range(3):
                cells[:, dx] += offsets[:, dx] * ncps

            offset_cells = cells.copy()
            offset_cells[:, 0] -= self.local_offset[2]
            offset_cells[:, 1] -= self.local_offset[1]
            offset_cells[:, 2] -= self.local_offset[0]

            offsets[:] = 1
            # check the new fmm cells are valid
            offsets[offset_cells[:, 0] < la[0], 0] = 0
            offsets[offset_cells[:, 0] > ua[0], 0] = 0
            offsets[offset_cells[:, 1] < la[1], 1] = 0
            offsets[offset_cells[:, 1] > ua[1], 1] = 0
            offsets[offset_cells[:, 2] < la[2], 2] = 0
            offsets[offset_cells[:, 2] > ua[2], 2] = 0
            if not np.all(offsets):
                raise RuntimeError("bad new cells")

            return cells, positions, shift_pos

    def _get_cell(self, position):
        # produces xyz tuple
        extent = self.group.domain.extent
        ncps = 2 ** (self.fmm.R - 1)
        cell_widths = [extent[0] / ncps, extent[1] / ncps, extent[2] / ncps]

        # convert to xyz
        ua = self.upper_allowed
        la = self.lower_allowed

        # compute position if origin was lower left not central
        cell = [
            min(int((0.5 * extent[0] + position[0]) / cell_widths[0]), ncps - 1),
            min(int((0.5 * extent[1] + position[1]) / cell_widths[1]), ncps - 1),
            min(int((0.5 * extent[2] + position[2]) / cell_widths[2]), ncps - 1),
        ]

        if self.boundary_condition is BCType.FREE_SPACE:
            # Proposed cell should never be over a periodic boundary, as there
            # are none.
            # Truncate down if too high on axis, if way too high this should
            # probably throw an error.
            return cell, np.array((0.0, 0.0, 0.0), dtype=REAL)
        else:
            assert self.boundary_condition in (BCType.PBC, BCType.NEAREST, BCType.FF_ONLY)
            # we assume that in both 27 nearest and pbc a proposed move could
            # be over a periodic boundary
            # following the idea that a proposed move is always in the
            # simulation domain we need to shift
            # positions accordingly
            spos = [0.5 * ex + po for po, ex in zip(position, extent)]
            # correct for round towards zero
            rtzc = [-1 if px < 0 else 0 for px in spos]
            cell = [cx + rx for cx, rx in zip(cell, rtzc)]

            offset = [((1 if cx < lx else 0) if cx <= ux else -1) for lx, cx, ux in zip(la, cell, ua)]

            # use the offsets to attempt to map into the region this rank has
            # data over the boundary
            cell = [cx + ox * ncps for cx, ox in zip(cell, offset)]
            lc = [cx >= lx for cx, lx in zip(cell, la)]
            uc = [cx <= ux for cx, ux in zip(cell, ua)]
            if not (all(lc) and all(uc)):
                raise RuntimeError(
                    "Could not map position into sub-domain. \
                    Check all proposed positions are valid"
                )

            return cell, np.array([ox * ex for ox, ex in zip(offset, extent)], dtype=REAL)
