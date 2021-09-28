# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
r'''

Angular power spectra transformations for Gaussian fields  (:mod:`gaussiancl`)
==============================================================================

This is a minimal Python package for working with the angular power spectra of
spherical random fields constructed from transformed Gaussian random fields.
It can currently transform power spectra of lognormal fields.

The package can be installed using pip::

    pip install gaussiancl

Then import the :func:`~gaussiancl.lognormal_cl` and
:func:`~gaussiancl.lognormal_normal_cl` functions from the package::

    from gaussiancl import lognormal_cl, lognormal_normal_cl

Current functionality covers the absolutely minimal use case.  Please open an
issue on GitHub if you would like to see anything added.


Distributions
-------------

Lognormal
~~~~~~~~~

.. math::

    Y = e^X - \lambda

.. math::

    \alpha = {\rm E}[Y] + \lambda


Reference/API
-------------

.. autosummary::
   :toctree: api
   :nosignatures:

   lognormal_cl
   lognormal_normal_cl

'''

__version__ = '2021.9.28'

__all__ = [
    'lognormal_cl',
    'lognormal_normal_cl',
]


import numpy as np
from transformcl import cltoxi, xitocl


def lognormal_cl(cl, alpha, alpha2=None, *, inv=False):
    '''lognormal angular power spectrum

    '''
    if alpha2 is None:
        alpha2 = alpha

    xi = cltoxi(cl)
    if not inv:
        xi /= alpha
        xi /= alpha2
        np.log1p(xi, out=xi)
    else:
        np.expm1(xi, out=xi)
        xi *= alpha
        xi *= alpha2
    return xitocl(xi)


def lognormal_normal_cl(cl, alpha, *, inv=False):
    '''lognormal cross normal angular power spectrum

    '''
    xi = cltoxi(cl)
    if not inv:
        xi /= alpha
    else:
        xi *= alpha
    return xitocl(xi)
