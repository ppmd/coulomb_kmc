from enum import Enum
from ppmd import opt

PROFILE = opt.PROFILE


class BCType(Enum):
    PBC = 'pbc'
    FREE_SPACE = 'free_space'
    NEAREST = '27'


