from enum import Enum
from ppmd import opt
import numpy as np

PROFILE = opt.PROFILE

def spherical(xyz):
    """
    Converts the cartesian coordinates in xyz to spherical coordinates
    (radius, polar angle, longitude angle)
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
    PBC = 'pbc'
    FREE_SPACE = 'free_space'
    NEAREST = '27'




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


