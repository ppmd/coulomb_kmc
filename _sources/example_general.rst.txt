**propose_with_dats**
=====================

.. highlight:: python


This is a skeleton example based on ``examples/propose_with_dats_example.py``, it is intended to provide the motivating ideas for **propose_with_dats**. For a executable script please use the example.

First we create a state ``A`` with the ``ParticleDats`` and ``ScalarArrays`` required.

::

    # Create a state "A"

    A = state.State()
    
    # Set the number of particles
    A.npart = N

    # Set the domain (must be cubic)
    A.domain = domain.BaseDomainHalo(extent=(E,E,E))
    # set the boundary condition for the domain (note this does not
    # set the domain for the FMMKMC)
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()
    
    # Create a PositionDat for particle positions and a ParticleDat
    # for particle charge values
    A.P = PositionDat(ncomp=3)
    A.Q = ParticleDat(ncomp=1)

    # Create dats for other desired properties such as global id
    A.GID = ParticleDat(ncomp=1, dtype=INT64)
    
    # ParticleDat for ``current_sites``
    A.sites = ParticleDat(ncomp=1, dtype=INT64)
    # ParticleDat for ``prop_positions``
    A.prop_positions = ParticleDat(ncomp=M*3)
    # ParticleDat for ``prop_masks``
    A.prop_masks = ParticleDat(ncomp=M, dtype=INT64)
    # ParticleDat for ``prop_energy_diffs``
    A.prop_diffs = ParticleDat(ncomp=M)

    # ScalarArray that holds the number of moves per site type
    # this example uses a cubic lattice with one site type,
    site_max_counts = ScalarArray(ncomp=1, dtype=INT64)


The charge properties should now be initialised. After particle positions and charges are set the ``KMCFMM`` instance can be created and initialised. The number of FMM levels ``r`` should be tuned for a particular machine and number of charges. The number of expansion terms ``l`` determines the accuracy of the method, e.g. ``l=12`` for ~4-5 significant figures.

::
    
    # create KMCFMM instance
    kmc_fmm = KMCFMM(positions=A.P, charges=A.Q, domain=A.domain, r=R,
    l=L, boundary_condition='pbc', max_move=max_move_dim)

    # initialise the KMCFMM instance
    kmc_fmm.initialise()


Algorithm 2 in  *Fast electrostatic solvers for kinetic Monte Carlo simulations* describes a sufficient method to populate the ``prop_positions`` and ``prop_masks`` ParticleDats.
See the :ref:`Bookkeeping` section for more details. The ``propose_with_dats_example.py`` example contains an implementation of this algorithm. Note: this algorithm (and implementation) can be readily extended to only consider charges that are close to an accepted move, this reduces the number of redundant bookkeeping operations.
After this bookkeeping is performed, the change in system energy for the set of all proposed moves is computed by the call to ``propose_with_dats``.

The ``KMCFMM`` implementation expects all proposed positions to be inside the simulation domain.

::

    kmc_fmm.propose_with_dats(site_max_counts, A.sites, A.prop_positions,
        A.prop_masks, A.prop_diffs, diff=True)


The computed energy difference are stored in the ``A.prop_diffs`` ParticleDat. By using a ParticleLoop or by using NumPy calls rates can be computed. Once a move is chosen to be accepted it should be passed to the ``accept``  method of the ``KMCFMM`` instance. This call will update the ``KMCFMM`` instance and update the particle position in the PositionDat the ``KMCFMM`` instance was created with.

The ``accept`` method is called with a tuple ``(id, np.array((r_x, r_y, r_z)))`` of local particle id and new position. By "local particle id" we require the local index of the particle to move on the owning MPI rank. In the provided example we use the ``np.where`` to locate a particle using its global id stored in ``A.GID``.

::

    kmc_fmm.accept(move)

Only one MPI rank should call ``accept`` with a tuple, all other ranks should pass ``None``.



