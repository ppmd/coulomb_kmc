Bookkeeping
===========

.. highlight:: python

Here we describe how to implement Algorithm 2 in  *Fast electrostatic solvers for kinetic Monte Carlo simulations* with a ParticleLoop and a PairLoop.
The ParticleLoop and PairLoop are the fundamental looping operations of the performance-portable framework for Molecular Dynamics (PPMD) https://github.com/ppmd/ppmd.
We use the following definitions for ParticleLoop and PairLoop.

:: 

    ParticleLoop = loop.ParticleLoopOMP
    PairLoop = pairloop.CellByCellOMP

We assume a state ``A`` is initialised as follows, as described in :ref:`**propose_with_dats**`.

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
    

The first step of the algorithm is to loop over all particles and populate the ParticleDat ``prop_positions`` with a ParticleLoop.
In this example the particles move on a cubic lattice.
The set of proposed moves are computed by taking the current particle position and adding an offset for each proposed position. In this example there are ``M`` proposed positions for each site (and ``M`` corresponding offsets).

::

    # make proposed positions kernel
    prop_pos_kernel_src = r'''
    // current pos
    const double p0 = P.i[0];
    const double p1 = P.i[1];
    const double p2 = P.i[2];

    // reset mask
    for (int mx=0 ; mx<M ; mx++){
        MASK.i[mx] = 1;
    }

    // form proposed positions from offsets
    for (int mx=0 ; mx< M ; mx++){
        double n0 = p0 + OA[mx*3 + 0];
        double n1 = p1 + OA[mx*3 + 1];
        double n2 = p2 + OA[mx*3 + 2];
        
        // wrap positions into domain
        if ( n0 < LOWER ) { n0 += EXTENT; } 
        if ( n1 < LOWER ) { n1 += EXTENT; } 
        if ( n2 < LOWER ) { n2 += EXTENT; } 
        if ( n0 > UPPER ) { n0 -= EXTENT; } 
        if ( n1 > UPPER ) { n1 -= EXTENT; } 
        if ( n2 > UPPER ) { n2 -= EXTENT; } 

        PP.i[mx*3 + 0] = n0;
        PP.i[mx*3 + 1] = n1;
        PP.i[mx*3 + 2] = n2;
    }
    '''

    # create a kernel from the kernel source
    prop_pos_kernel = kernel.Kernel(
        'prop_pos_kernel', 
        prop_pos_kernel_src, 
        constants=(
            Constant('M', M),
            Constant('LOWER', -0.5 * E),
            Constant('UPPER', 0.5 * E),
            Constant('EXTENT', E)
        )
    )

    # create a ParticleLoop from the kernel
    prop_pos = ParticleLoop(
        kernel=prop_pos_kernel, 
        dat_dict={
            'P'     : A.P(READ),
            'PP'    : A.prop_positions(WRITE),
            'OA'    : offsets_sa(READ),
            'MASK'  : A.prop_masks(WRITE)
        }
    )

The ParticleLoop ``prop_pos`` is executed by calling

::

    prop_pos.execute()


Now that the proposed positions are stored in ``prop_pos`` we remove proposed moves that would result in particle overlap by masking off "bad" moves.
We discover potential overlaps with the PairLoop operation.
For each particle we loop over neighbouring particles within ``max_move`` and check for an overlap with the proposed moves.
If an overlap is detected then the corresponding move is removed by setting ``prop_masks`` to a value less than 1 for that move.
We allow for any value less than 1 such that recombination events can be identified, the current FMM-KMC implementation does not support recombination.

We provide now the example kernel to mask off proposed moves that would cause an overlap. It is important to note that PPMD **does not** wrap particle positions around the periodic boundary.
To be explicit: if a particle has a :math:`x` position of :math:`-0.5E + \delta` then it will be exposed with a :math:`x` position of :math:`0.5E + \delta` when observed over the periodic boundary in a PairLoop.
This convention avoids the use of conditionals in PairLoops for inter-particle interactions, which is the primary use case of the PPMD framework.

Bearing this in mind, we recompute the proposed moves for each particle for each neighbour, without wrapping around the boundary, and then compare particle positions.

::

    # make exclude kernel

    exclude_kernel_src = r'''
    // current position of particle i
    const double p0 = P.i[0];
    const double p1 = P.i[1];
    const double p2 = P.i[2];

    // read particle j position
    const double pj0 = P.j[0];
    const double pj1 = P.j[1];
    const double pj2 = P.j[2];

    // check each proposed position
    for (int mx=0 ; mx< M ; mx++){

        // form the proposed position without wrapping
        double n0 = p0 + OA[mx*3 + 0];
        double n1 = p1 + OA[mx*3 + 1];
        double n2 = p2 + OA[mx*3 + 2];
        
        // difference to particle j position
        const double d0 = pj0 - n0;
        const double d1 = pj1 - n1;
        const double d2 = pj2 - n2;
        
        // if they overlap, mask off the position
        const double r2 = d0*d0 + d1*d1 + d2*d2;

        // using a ternary operator aids autovectorisation.
        MASK.i[mx] = (r2 < TOL) ? 0 : MASK.i[mx];
    }
    '''

    # create kernel from kernel source
    exclude_kernel = kernel.Kernel(
        'exclude_kernel', 
        exclude_kernel_src, 
        constants=(
            Constant('M', M),
            Constant('TOL', 0.01)
        )
    )

    # create pairloop from kernel
    exclude = PairLoop(
        kernel=exclude_kernel, 
        dat_dict={
            'P'     : A.P(READ),
            'OA'    : offsets_sa(READ),
            'MASK'  : A.prop_masks(WRITE)
        },
        shell_cutoff = max_move
    )


As for the ParticleLoop, the PairLoop ``exclude`` can be executed with

::
    
    exclude.execute()


