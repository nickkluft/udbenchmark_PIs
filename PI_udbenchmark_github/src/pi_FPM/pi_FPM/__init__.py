from .version import __version__
from .footplacementmodel import calcFPM, timenormalize_step
# if somebody does "from somepackage import *", this is what they will
# be able to access:
__all__ = ['calcFPM',
           'timenormalize_step']

