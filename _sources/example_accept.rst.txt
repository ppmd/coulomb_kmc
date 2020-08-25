**accept**
==========

.. highlight:: python

To accept a move in a ``KMCFMM`` we provide the ``accept`` method.
This methods is passed a tuple ``(id, position)`` where ``id`` is a particle id local to the calling MPI rank and ``position`` is the new position of the particle.

This method is a collective call, all other MPI ranks should pass ``None`` to ``accept``.
The ``KMCFMM`` instance will update the position of the particle with local id `id` in the PositionDat that it was initialised with.
Note, the new particle position could be in a subdomain owned by an MPI rank that is different to the MPI rank that calls ``accept``.
In this scenario the particle will move MPI ranks when ``accept`` is called.

For example

::
    
    kmc_fmm.accept((0, np.array(1.1, 1.2, 3.0)))



