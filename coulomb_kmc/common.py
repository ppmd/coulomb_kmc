from enum import Enum
import numpy as np

from ppmd.opt import PROFILE as PROFILE

def spherical(xyz):
    """
    Converts the cartesian coordinates in xyz to spherical coordinates
    (radius, polar angle, longitude angle)
    
    :arg xyz: Input xyz coordinates as Numpy array or tuple/list.
    """
    if type(xyz) is tuple:
        sph = np.zeros(3)
        xy = xyz[0]**2 + xyz[1]**2
        # r
        sph[0] = np.sqrt(xy + xyz[2]**2)
        # polar angle
        sph[1] = np.arctan2(np.sqrt(xy), xyz[2])
        # longitude angle
        sph[2] = np.arctan2(xyz[1], xyz[0])

    else:
        sph = np.zeros(xyz.shape)
        xy = xyz[:,0]**2 + xyz[:,1]**2
        # r
        sph[:,0] = np.sqrt(xy + xyz[:,2]**2)
        # polar angle
        sph[:,1] = np.arctan2(np.sqrt(xy), xyz[:,2])
        # longitude angle
        sph[:,2] = np.arctan2(xyz[:,1], xyz[:,0])

    return sph


class BCType(Enum):
    """
    Enum to indicate boundary condition type.
    """

    PBC = 'pbc'
    """Fully periodic boundary conditions"""
    FREE_SPACE = 'free_space'
    """Free-space, e.g. vacuum, boundary conditions."""
    NEAREST = '27'
    """Primary image and the surrounding 26 nearest neighbours."""
    FF_ONLY = 'ff-only'
    """Only the far-field contribution. I.E. 'pbc' without '27'"""


cell_offsets = (
    ( -1, -1, -1),
    (  0, -1, -1),
    (  1, -1, -1),
    ( -1,  0, -1),
    (  0,  0, -1),
    (  1,  0, -1),
    ( -1,  1, -1),
    (  0,  1, -1),
    (  1,  1, -1),

    ( -1, -1,  0),
    (  0, -1,  0),
    (  1, -1,  0),
    ( -1,  0,  0),
    (  0,  0,  0),
    (  1,  0,  0),
    ( -1,  1,  0),
    (  0,  1,  0),
    (  1,  1,  0),

    ( -1, -1,  1),
    (  0, -1,  1),
    (  1, -1,  1),
    ( -1,  0,  1),
    (  0,  0,  1),
    (  1,  0,  1),
    ( -1,  1,  1),
    (  0,  1,  1),
    (  1,  1,  1),
)

cell_offsets_26 = (
    ( -1, -1, -1),
    (  0, -1, -1),
    (  1, -1, -1),
    ( -1,  0, -1),
    (  0,  0, -1),
    (  1,  0, -1),
    ( -1,  1, -1),
    (  0,  1, -1),
    (  1,  1, -1),

    ( -1, -1,  0),
    (  0, -1,  0),
    (  1, -1,  0),
    ( -1,  0,  0),
    (  1,  0,  0),
    ( -1,  1,  0),
    (  0,  1,  0),
    (  1,  1,  0),

    ( -1, -1,  1),
    (  0, -1,  1),
    (  1, -1,  1),
    ( -1,  0,  1),
    (  0,  0,  1),
    (  1,  0,  1),
    ( -1,  1,  1),
    (  0,  1,  1),
    (  1,  1,  1),
)


def add_flop_dict(d1, d2):
    """
    Combines two FLOP counting dicts.
    """
    for kx in ('+', '-', '*', '/'):
        d1[kx] += d2[kx]
    return d1

def new_flop_dict():
    """
    Returns a new FLOP counting dict with zero counts.
    """
    return {'+': 0, '-': 0, '*': 0, '/': 0}


class ProfInc:
    """
    Base class to inherit that gives profiling methods.
    """
    def _profile_inc(self, key, inc):
        key = self.__class__.__name__ + ':' + key
        if key not in PROFILE.keys():
            PROFILE[key] = inc
        else:
            PROFILE[key] += inc

    def _profile_get(self, key):
        key = self.__class__.__name__ + ':' + key
        return PROFILE[key]

    def _profile_set(self, key, inc):
        key = self.__class__.__name__ + ':' + key
        if key not in PROFILE.keys():
            PROFILE[key] = inc
        else:
            PROFILE[key] = inc











