Development Guide
=================

The important dependencies between the Python modules is illustrated in the following graph:

.. graphviz::

   digraph {
        node [shape=box,style=solid];

        kmc_dirichlet_boundary;

        kmc_fmm -> kmc_local;
        kmc_fmm -> kmc_full_long_range;
        kmc_fmm -> kmc_octal;
        kmc_fmm -> kmc_fmm_self_interaction;
        kmc_fmm -> kmc_mpi_decomp;
        kmc_fmm -> kmc_inject_extract;

        kmc_fmm                -> kmc_fmm_common;
        kmc_local              -> kmc_fmm_common;
        kmc_full_long_range    -> kmc_fmm_common;
        kmc_octal              -> kmc_fmm_common;
        kmc_mpi_decomp         -> kmc_fmm_common;

        kmc_inject_extract -> kmc_direct;

        kmc_fmm                -> kmc_expansion_tools;
        kmc_direct             -> kmc_expansion_tools;
        kmc_octal              -> kmc_expansion_tools;
        kmc_inject_extract     -> kmc_expansion_tools;
        
   }


We now give a brief description of the functionality provided in each module.

kmc_fmm
-------

Provides the main ``KMCFMM`` class which, by default, defines the ``initialise``, ``propose``, ``propose_with_dats`` and ``accept`` interfaces. 
When ``initialise`` is called a FMM solver from ``PPMD`` is created and called to compute the initial system energy and construct the initial local expansions.
A instance of ``FMMMPIDecomp`` is created to handle the MPI decomposition of the domain.
As part of the initialisation process the data structures for direct interactions are initialised using ``kmc_local`` along with data structures for indirect interactions using ``kmc_octal``. 
Long range interactions are initialised using ``kmc_full_long_range`` and the self interactions are initialised using the ``kmc_self_interaction`` module.


Periodic boundary conditions are enacted by splitting the domain into two sets of periodic images.
The first is the inner set which is formed from the primary image and its nearest neighbours.
In the inner set energy differences from proposed moves are computed by considering the change in potential field of a move.
i.e. the difference is calculated in both the indirect and direct parts, as opposed to computing the new energy of the new configuration and subtracting the old energy.
In the second "far-field" set of periodic images, the new energy is computed in full for each proposed move.
This splitting approach is why the FMM solver from ``PPMD`` is used in "27" mode even for the fully periodic case.

kmc_dirichlet_boundary
----------------------

This is a helper module contains the class ``MirrorChargeSystem`` that aids the construction of systems where each input charge is mirrored in the z-direction such that the potential field is zeroed on the z boundaries.

kmc_mpi_decomp
--------------

The ``FMMMPIDecomp`` class performs a domain decomposition of the domain over the MPI ranks. This decomposition step determines which local expansions are required by each rank by taking into account the maximum move size. This class also provides the ``setup_propose`` method which translates passed proposed moves, in either the ``propose`` or ``propose_with_dats`` format, into an internal format that can be passed to the modules that handle the indirect, direct, long-range and self interaction components.

kmc_octal
---------

The class ``LocalCellExpansions`` handles the interactions through local expansions for the primary image and its nearest 26 neighbours. At initialisation the required expansions are copied from the owning rank in the FMM solver using MPI RMA operations.
Proposed moves are passed using the format specified in ``kmc_mpi_decomp``.


kmc_local
---------

The class ``LocalParticleData`` computes direct interactions between charges. At initialisation particle positions and charges are copied into local data structures. Proposed moves are passed in the standard format specified in ``kmc_mpi_decomp``.

kmc_fmm_self_interaction
------------------------

When moves are proposed the new energy contains a self interaction contribution that should be subtracted. This module computes the self interactions between the new position and the old images in the primary image and its nearest neighbours. There are no spurious contributions from further away periodic images as these images are handled "in full" by the ``kmc_full_long_range`` module.

kmc_full_long_range
-------------------

Computes the far-field contribution for proposed moves (and initialisation). The main class ``FullLongRangeEnergy`` maintains the level zero multipole expansion for the original charge locations in the domain. When a move is proposed the current contribution for the moving charge is subtracted and the new contribution is added. The long range multipole to local operator is applied to give the local expansion for the far-field contribution.

The class also maintains the vector of coefficients that when used in a dot product with the far-field local expansion give the far-field energy contribution. As with the multipole expansion, the current contribution of the moving charge is subtracted from this vector and the new contribution added. The new set of coefficients is then used to compute the proposed far-field contribution.

kmc_inject_extract
------------------

Contains a parent class that when inherited adds the ``propose_inject``, ``inject``, ``propose_extract`` and ``extract`` methods to the ``KMCFMM`` class. When extractions/injections are proposed this class uses functionality in ``kmc_direct`` to compute the self energy of the involved group of charges. In general, this self energy is non-zero when the group of charges involved contains more than one charge. As we only allow injections and extractions in mirror mode there is always a self energy term.

This module also contains the ``DiscoverInjectExtract`` class which, at each iteration,  finds empty inject sites and charges that exist on extract sites. Critically, these methods assume that the number of inject and extract sites is small in comparison to the number of charges in the simulation.

kmc_direct
----------

Contains electrostatic solvers for systems containing small numbers of charges. i.e. small enough that direct methods are the most efficient to compute the electrostatic energy of the system. Along with testing, these methods are used to compute the self energy of groups of charges in inject and extract scenarios.


kmc_expansion_tools
-------------------

Contains methods to generate code that computes and evaluates local and multipole expansions. This functionality should (and is) be merged into ``PPMD``.




