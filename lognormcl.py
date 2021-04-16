# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''

Angular power spectra of lognormal fields  (:mod:`lognormcl`)
=============================================================

This is a minimal Python package for working with the angular power spectra of
lognormal spherical random fields.  It can currently convert between the power
spectra of lognormal random fields and their constituent normal random fields.

The package can be installed using pip::

    pip install lognormcl

Then import the :func:`~lognormcl.ln2n` and :func:`~lognormcl.n2ln` functions
from the package::

    from lognormcl import ln2n, n2ln

Current functionality covers the absolutely minimal use case.  Please open an
issue on GitHub if you would like to see anything added.


Reference/API
-------------

.. autosummary::
   :toctree: api
   :nosignatures:

   ln2n
   n2ln

'''

__version__ = '2021.4.16'

__all__ = [
    'ln2n',
    'n2ln',
]


import numpy as np
from transformcl import cltoxi, xitocl


def ln2n(cl, alpha, alpha2=None):
    '''lognormal to normal angular power spectrum
    '''

    xi = cltoxi(cl)
    xi /= alpha
    xi /= alpha2 if alpha2 is not None else alpha
    np.log1p(xi, out=xi)

    return xitocl(xi)



def n2ln(cl, alpha, alpha2=None):
    '''normal to lognormal angular power spectrum
    '''

    xi = cltoxi(cl)
    np.expm1(xi, out=xi)
    xi *= alpha
    xi *= alpha2 if alpha2 is not None else alpha

    return xitocl(xi)
