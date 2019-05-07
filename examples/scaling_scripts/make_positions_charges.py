from __future__ import print_function, division

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"

import ctypes
import math
import numpy as np
import sys
import pickle

from ppmd import utility


REAL = ctypes.c_double



if __name__ == '__main__':
    
    N = int(sys.argv[1])
    Ns = int(math.ceil(N**(1./3.)))
    E = 3.3 * Ns

    positions = utility.lattice.cubic_lattice((Ns, Ns, Ns), (E, E, E))
    positions = positions[:N, :]
        
    charges = np.zeros(N, REAL)
    for cx in range(N):
        charges[cx] = (-1.0) ** cx

    bias = np.sum(charges) / N
    charges -= bias
    

    sim_data = {
        'P': positions,
        'Q': charges,
        'E': E
    }

    name = str(sys.argv[2])
    
    with open(name, 'wb') as fh:
        pickle.dump(sim_data, fh)





