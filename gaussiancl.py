# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
r'''

Angular power spectra transformations for Gaussian fields  (:mod:`gaussiancl`)
==============================================================================

This is a minimal Python package for working with the angular power spectra of
spherical random fields constructed by transforming Gaussian random fields.

It can solve for the Gaussian angular power spectrum that will yield a given
target angular power spectrum under a given transformation, such as lognormal.

The package can be installed using pip::

    pip install gaussiancl

Then import the :func:`~gaussiancl.gaussiancl` function from the package::

    from gaussiancl import gaussiancl

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

   gcllim
   gaussiancl
   lognormal
   lognormal_normal

'''

__version__ = '2023.1'

__all__ = [
    'gcllim',
    'gaussiancl',
    'lognormal',
    'lognormal_normal',
]


import numpy as np
from transformcl import cltocorr, corrtocl


def _gettfm(name):
    '''get a transform by name'''
    g = globals()
    if name not in g:
        raise ValueError(f'unknown transformation: {name}')
    return g[name]


def _relerr(dx, x):
    '''compute the relative error max(|dx/x|)'''
    q = np.divide(dx, x, where=(dx != 0), out=np.zeros_like(dx))
    return np.fabs(q).max()


def gcllim(cl, tfm, pars=(), *, inv=False, der=False):
    '''band-limited Gaussian cl transform'''
    if not callable(tfm):
        tfm = _gettfm(tfm)
    return corrtocl(tfm(cltocorr(cl), *pars, inv=inv, der=der))


def gaussiancl(cl, tfm, pars=(), *, gl=None, n=None, cltol=1e-5, gltol=1e-5,
               maxiter=20, monopole=None):
    '''Solve for a Gaussian angular power spectrum.

    The function returns a status indicator ``info`` as follows:

    * ``0``, solution did not converge using the maximum number of iterations;
    * ``1``, solution converged in cl relative error;
    * ``2``, solution converged in gl relative error;
    * ``3``, solution converged in both of the above.

    Returns
    -------
    gl : array_like
        Gaussian angular power spectrum solution.
    info : int
        Status indicator.
    clerr : float
        Achieved relative error in the angular power spectrum.
    niter : int
        Number of iterations.

    '''

    if not callable(tfm):
        tfm = _gettfm(tfm)

    m = len(cl)
    if n is not None:
        if not isinstance(n, int):
            raise TypeError('n must be integer')
        elif n < m:
            raise ValueError('n must be larger than the length of cl')
        k = n - m
    else:
        k = 2*len(cl)

    if gl is None:
        gl = corrtocl(tfm(cltocorr(cl), *pars, inv=True))
    elif monopole is not None:
        gl = np.copy(gl)
    if monopole is not None:
        gl[0] = monopole

    gt = cltocorr(np.pad(gl, (0, k)))
    fl = corrtocl(tfm(gt, *pars))[:m] - cl
    if monopole is not None:
        fl[0] = 0
    clerr = _relerr(fl, cl)

    info = 0
    for i in range(maxiter):
        if clerr <= cltol:
            info |= 1
        if info > 0:
            break

        ft = cltocorr(np.pad(fl, (0, k)))
        dt = tfm(gt, *pars, der=True)
        xl = -corrtocl(ft/dt)[:m]
        if monopole is not None:
            xl[0] = 0

        while True:
            gl_ = gl + xl
            gt_ = cltocorr(np.pad(gl_, (0, k)))
            fl_ = corrtocl(tfm(gt_, *pars))[:m] - cl
            if monopole is not None:
                fl_[0] = 0
            clerr_ = _relerr(fl_, cl)
            if clerr_ <= clerr:
                break
            xl /= 2

        if _relerr(xl, gl) <= gltol:
            info |= 2

        gl, gt, fl, clerr = gl_, gt_, fl_, clerr_

    return gl, info, clerr, i


def lognormal(x, alpha, alpha2=None, *, inv=False, der=False):
    '''lognormal correlations'''
    if alpha2 is None:
        alpha2 = alpha
    if not inv:
        if not der:
            return np.expm1(x)*(alpha*alpha2)
        else:
            return np.exp(x)*(alpha*alpha2)
    else:
        if not der:
            return np.log1p(x/(alpha*alpha2))
        else:
            return 1/(x + alpha*alpha2)


def lognormal_normal(x, alpha, *, inv=False, der=False):
    '''lognormal cross normal correlations'''
    if not inv:
        if not der:
            return x*alpha
        else:
            return alpha
    else:
        if not der:
            return x/alpha
        else:
            return 1/alpha
