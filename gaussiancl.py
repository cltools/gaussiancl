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
   normal
   lognormal
   lognormal_normal

'''

__version__ = '2022.10.14'

__all__ = [
    'gcllim',
    'gaussiancl',
    'normal',
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


def gcllim(cl, tfm, pars=(), *, inv=False, der=False):
    '''band-limited Gaussian cl transform'''
    if not callable(tfm):
        tfm = _gettfm(tfm)
    return corrtocl(tfm(cltocorr(cl), *pars, inv=inv, der=der))


def gaussiancl(cl, tfm, pars=(), *, gl=None, n=None, tol=1e-5, maxiter=20,
               monopole=None, return_iter=False, return_tol=False):
    '''Solve for a Gaussian angular power spectrum.

    The function returns an error indicator ``err`` as follows:

    * ``0``, converged solution, no error;
    * ``-i``, solution did not converge but no further improvement was obtained
      after ``i`` iterations;
    * ``i``, solution did not converge in the maximum number of ``i``
      iterations.

    This means that ``err > 0`` checks if the solution could be improved.

    Returns
    -------
    gl : array_like
        Gaussian angular power spectrum solution.
    err : int
        Error indicator.

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
    f2 = np.dot(fl, fl)

    i = 0
    while np.any(np.fabs(fl) > tol*np.fabs(cl)):
        i += 1
        if i > maxiter:
            i -= 1
            err = i
            break

        ft = cltocorr(np.pad(fl, (0, k)))
        dt = tfm(gt, *pars, der=True)
        xl = -corrtocl(ft/dt)[:m]
        if monopole is not None:
            xl[0] = 0

        while np.dot(xl, xl) != 0:
            gl_ = gl + xl
            gt_ = cltocorr(np.pad(gl_, (0, k)))
            fl_ = corrtocl(tfm(gt_, *pars))[:m] - cl
            if monopole is not None:
                fl_[0] = 0
            f2_ = np.dot(fl_, fl_)
            if f2_ <= f2:
                break
            xl = xl/2
        else:
            err = -i
            break

        gl, gt, fl, f2 = gl_, gt_, fl_, f2_
    else:
        err = 0

    result = gl, err

    if return_iter:
        result = *result, i
    if return_tol:
        result = *result, np.fabs(fl[fl != 0]/cl[fl != 0]).max()

    return result


def normal(x, *, inv=False, der=False):
    '''normal correlations'''
    if not der:
        return x
    else:
        return np.zeros_like(x)


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
