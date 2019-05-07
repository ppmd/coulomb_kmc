Bookkeeping
===========

.. highlight:: python

Here we describe how to implement Algorithm 2 in  *Fast electrostatic solvers for kinetic Monte Carlo simulations* with a ParticleLoop and a PairLoop. We use the following definitions for ParticleLoop and PairLoop.

:: 

    ParticleLoop = loop.ParticleLoopOMP
    PairLoop = pairloop.CellByCellOMP

We assume a state ``A`` is initialised as follows, as described in :ref:`**propose_with_dats** Example`.

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
    







