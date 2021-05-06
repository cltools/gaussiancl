# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
r'''

Angular power spectra conversions for normal fields  (:mod:`normcl`)
====================================================================

This is a minimal Python package for working with the angular power spectra of
spherical random fields constructed from normals.  It can currently convert
between the power spectra of lognormal random fields and their constituent
normal random fields.

The package can be installed using pip::

    pip install normcl

Then import the :func:`~normcl.lognormal` and :func:`~normcl.lognormal_normal`
functions from the package::

    from normcl import lognormal, lognormal_normal

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

   lognormal
   lognormal_normal

'''

__version__ = '2021.5.5'

__all__ = [
    'lognormal',
    'lognormal_normal',
]


import numpy as np
from transformcl import cltoxi, xitocl


def lognormal(cl, alpha, alpha2=None, *, inv=True):
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


def lognormal_normal(cl, alpha, *, inv=False):
    '''lognormal cross normal angular power spectrum

    '''
    xi = cltoxi(cl)
    if not inv:
        xi /= alpha
    else:
        xi *= alpha
    return xitocl(xi)
