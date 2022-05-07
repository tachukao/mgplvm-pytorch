from . import kernels
from . import priors
from . import models
from . import optimisers
from . import lat_dist
from . import manifolds
from . import crossval
from . import syndata
from . import fast_utils

from .lat_dist import (ReLie, LatentDistribution, GPBaseLatDist, GPCircLatDist, GPDiagLatDist)
from .kernels import Linear as LinearKernel
from .kernels import QuadExp as QuadExpKernel
from .kernels import Exp as ExpKernel
from .kernels import Matern as MaternKernel
