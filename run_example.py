#!/usr/bin/env python3
'''
Be star's polarization using McDavid approximation
'''
from pol import *

# nicer plot font
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['font.size'] = 14

# input parameters
N0e = 9e12
i = 57.6
lmin, lmax, nl = 1000, 10000, 100
lbd = np.linspace(lmin, lmax, nl)
lbd *= 1e-8  # angstrom to cm
lbdc = .545e-4 # desired wavelength

# stellar par
logg = 4.17 # superficial gravity
Teff = 25940 # effective temperature
M = 11.03  # stellar mass
Rs = 5.11  # equatorial radius
oblat = 5.11 / 4.5 #oblatness

Rp = Rs / oblat
Td = .6 * Teff
H0 = (k * Td / mu / mh * (Rs * Rsol)**3 / G / (M * Msol))**.5

stellar_params = [Teff, Td, Rs, logg, i]

method = 0 #0 for mcdavid opacity, 1 for bjorkman opacity, 2 for mcdavid opacity with NLTE correction

r = np.linspace(1, 115, 399)
Sigma = 0.1 * r**(-1.1)

p = pol(r, Sigma, 10, 1.1, stellar_params, lbdc, method)
