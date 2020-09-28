from .version import __version__
from .localdivergence import calcLDE, calcStateSpace
# if somebody does "from somepackage import *", this is what they will
# be able to access:
__all__ = [
    'clacLDE',
    'calcStateSpace']
