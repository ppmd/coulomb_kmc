FMM-KMC Documentation
=====================

This is the documentation for the reference implementation of the FMM-KMC algorithm described in *Fast electrostatic solvers for kinetic Monte Carlo simulations*, https://arxiv.org/pdf/1905.04065.pdf.

There exists a main Python class ``FMMKMC`` that provides the interface to our implementation.
After initialisation, this class provides methods to propose and accept particle moves using the method we describe in the above paper.

This implementation supports MPI and OpenMP parallelism and is applicable to cubic systems with fully periodic, free space and Dirichlet (plate like, work in progress) boundary conditions.

We provide this implementation as an extension to the performance-portable Molecular Dynamics framework (PPMD): https://github.com/ppmd/ppmd. 

The source code will be made available at https://github.com/ppmd/coulomb_kmc.



.. toctree::
   :maxdepth: 2
   :caption: Contents:
    
   installation
   kmc_fmm
   example_general
   example_bookkeeping
   example_simple
   example_accept
 

Generated Documentation
=======================

.. toctree::
    
    modules/coulomb_kmc


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`




