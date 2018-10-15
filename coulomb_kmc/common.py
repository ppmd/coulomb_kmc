from enum import Enum

class BCType(Enum):
    PBC = 'pbc'
    FREE_SPACE = 'free_space'
    NEAREST = '27'


PROFILE = {}
"""
Dict available module wide for profiling. Recommended format along lines of:

{
    'description'
:
    (
        total_time_taken
    )
}
"""

def print_profile():
    for key, value in sorted(PROFILE.items()):
        print(key)
        print('\t', value)
