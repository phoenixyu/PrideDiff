from __future__ import division
import numpy as np
from typing import List, Union, Tuple

def phantom(
	n: Union[int, Tuple[int], List[int]] = 256, 
	p_type: str = 'Modified Shepp-Logan', 
	ellipses: Union[List, np.ndarray, None] = None, 
	dim: int = 2, 
	zlims: Union[Tuple[float], List[float]] = (-1., 1.)
):

	if np.isscalar(n):
		shape = tuple([n] * dim)
	else:
		shape = tuple(n)

	if (ellipses is None):
		ellipses = _select_phantom(p_type, dim)

	if (dim == 2):
		p = create_phantom2d(shape, ellipses)

	elif (dim == 3):
		p = create_phantom3d(shape, ellipses, zlims)

	else:
		raise ValueError("Unexpected dim: %d" % dim)

	return p


def create_phantom2d(shape, ellipses):
	assert len(shape) == 2, f"Unexpected dimension"
	assert np.ndim(ellipses) == 2, f"Wrong dims in user phantom"
	assert np.size(ellipses, 1) == 6, f"Wrong number of columns in user phantom"

	p = np.zeros(shape)
	m, n = shape
	ygrid, xgrid = np.mgrid[1:-1:(1j*m), -1:1:(1j*n)]

	for ellip in ellipses:
		I   = ellip[0]
		a2  = ellip[1]**2
		b2  = ellip[2]**2
		x0  = ellip[3]
		y0  = ellip[4]
		phi = ellip[5] * np.pi / 180

		x = xgrid - x0
		y = ygrid - y0

		cos_p = np.cos(phi)
		sin_p = np.sin(phi)

		locs = (((x * cos_p + y * sin_p)**2) / a2
			  + ((y * cos_p - x * sin_p)**2) / b2) <= 1

		p[locs] += I

	return p


def create_phantom3d(shape, ellipses, zlims):
	assert len(shape) == 3, f"Unexpected dimension"
	assert np.ndim(ellipses) == 2, f"Wrong dims in user phantom"
	assert np.size(ellipses, 1) == 8, f"Wrong number of columns in user phantom"
	assert len(zlims) == 2, f"zlims must be a tuple with 2 entries: upper and lower bounds"
	assert zlims[0] <= zlims[1], f"zlims: lower bound must be first entry"

	p = np.zeros(shape)
	m, n, d = shape
	ygrid, xgrid, zgrid = np.mgrid[1:-1:(1j*m), -1:1:(1j*n), zlims[0]:zlims[1]:(1j*d)]
	for ellip in ellipses:
		I   = ellip[0]
		a2  = ellip[1]**2
		b2  = ellip[2]**2
		c2  = ellip[3]**2
		x0  = ellip[4]
		y0  = ellip[5]
		z0  = ellip[6]
		phi = ellip[7] * np.pi / 180

		x = xgrid - x0
		y = ygrid - y0
		z = zgrid - z0

		cos_p = np.cos (phi)
		sin_p = np.sin (phi)

		locs = (((x * cos_p + y * sin_p)**2) / a2
			  + ((y * cos_p - x * sin_p)**2) / b2
			  + z ** 2 / c2) <= 1

		p[locs] += I

	return p

def _select_phantom(name, dim):
	if dim == 2:
		if (name.lower () == 'shepp-logan'):
			e = _shepp_logan()
		elif (name.lower () == 'modified shepp-logan'):
			e = _mod_shepp_logan()
		else:
			raise ValueError("Unknown phantom type: %s" % name)

	elif dim == 3:
		if (name.lower () == 'shepp-logan'):
			e = _shepp_logan_3d()
		elif (name.lower () == 'modified shepp-logan'):
			e = _mod_shepp_logan_3d()
		else:
			raise ValueError("Unknown phantom type: %s" % name)

	else:
		raise ValueError("Unexpected dim: %d" % dim)

	return e


def _shepp_logan():
	#  Standard head phantom, taken from Shepp & Logan
	return [[   2,   .69,   .92,    0,      0,   0],
			[-.98, .6624, .8740,    0, -.0184,   0],
			[-.02, .1100, .3100,  .22,      0, -18],
			[-.02, .1600, .4100, -.22,      0,  18],
			[ .01, .2100, .2500,    0,    .35,   0],
			[ .01, .0460, .0460,    0,     .1,   0],
			[ .02, .0460, .0460,    0,    -.1,   0],
			[ .01, .0460, .0230, -.08,  -.605,   0],
			[ .01, .0230, .0230,    0,  -.606,   0],
			[ .01, .0230, .0460,  .06,  -.605,   0]]


def _mod_shepp_logan():
	#  Modified version of Shepp & Logan's head phantom,
	#  adjusted to improve contrast.  Taken from Toft.
	return [[   1,   .69,   .92,    0,      0,   0],
			[-.80, .6624, .8740,    0, -.0184,   0],
			[-.20, .1100, .3100,  .22,      0, -18],
			[-.20, .1600, .4100, -.22,      0,  18],
			[ .10, .2100, .2500,    0,    .35,   0],
			[ .10, .0460, .0460,    0,     .1,   0],
			[ .10, .0460, .0460,    0,    -.1,   0],
			[ .10, .0460, .0230, -.08,  -.605,   0],
			[ .10, .0230, .0230,    0,  -.606,   0],
			[ .10, .0230, .0460,  .06,  -.605,   0]]


def _shepp_logan_3d():
	return [[   2,   .69,   .92,    .9,    0,     0,    0,   0],
			[-.80, .6624, .8740, .8800,    0,     0,    0,   0],
			[-.20, .4100, .1600, .2100, -.22,     0, -.25, 108],
			[-.20, .3100, .1100, .2200,  .22,     0, -.25,  72],
			[ .20, .2100, .2500, .5000,    0,   .35, -.25,   0],
			[ .20, .0460, .0460, .0460,    0,   .10, -.25,   0],
			[ .10, .0460, .0230, .0200, -.08,  -.65, -.25,   0],
			[ .10, .0460, .0230, .0200,  .06,  -.65, -.25,  90],
			[ .20, .0560, .0400, .1000,  .06, -.105, .625,  90],
			[-.20, .0560, .0560, .1000,    0,    .1, .625,   0]]


def _mod_shepp_logan_3d():
	return [[   1,   .69,   .92,    .9,    0,     0,    0,   0],
			[-.80, .6624, .8740, .8800,    0,     0,    0,   0],
			[-.20, .4100, .1600, .2100, -.22,     0, -.25, 108],
			[-.20, .3100, .1100, .2200,  .22,     0, -.25,  72],
			[ .10, .2100, .2500, .5000,    0,   .35, -.25,   0],
			[ .10, .0460, .0460, .0460,    0,   .10, -.25,   0],
			[ .10, .0460, .0230, .0200, -.08,  -.65, -.25,   0],
			[ .10, .0460, .0230, .0200,  .06,  -.65, -.25,  90],
			[ .10, .0560, .0400, .1000,  .06, -.105, .625,  90],
			[-.10, .0560, .0560, .1000,    0,    .1, .625,   0]]

