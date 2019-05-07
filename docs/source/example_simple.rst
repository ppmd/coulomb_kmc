**propose**
===========

.. highlight:: python

It is recommended that the :ref:`**propose_with_dats**` interface is used.
There is a simpler (and less efficient) method to propose moves called ``propose``.

Using ``propose`` proposed moves can be passed to a ``KMCFMM`` instance as a tuple of proposed moves.
If a ``KMCFMM`` instance is created as in :ref:`**propose_with_dats**`, then a set of proposed moves can be passed as

::

    energies = kmc_fmm.propose(
        (   
            id_0, np.array(((r_00x, r_00y, r_00z),
                            (r_01x, r_01y, r_01z)))
        ),
        (   
            id_1, np.array(((r_10x, r_10y, r_10z),
                            (r_11x, r_11y, r_11z)))
        )       
    )

The particle ids, e.g. `id_0`, are local to the calling MPI rank.
The call will return a tuple of NumPy arrays that correspond to the proposed moves that were passed.
These arrays contain the system energy that would result from the proposed move being accepted.




