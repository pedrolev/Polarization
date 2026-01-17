#!/usr/bin/env python3
'''
Be star's polarization using McDavid approximation
'''
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import griddata
import pyhdust.spectools as spt
import pyhdust.phc as phc

# nicer plot font
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['font.size'] = 14

# constants
el = 4.8032e-10
me = 9.1093837015e-28
mh = 1.672622e-24
k = 1.380650e-16
c = 29979245800
h = 6.626069e-27
R = 1.0967768e5
Msol = 1.989e33
Rsol = 6.955e10
mi = 0.6
Z = 1
G = 6.67430e-8

# input parameters
N = 3.1
N0e = 9e12
i = 57.6
lmin, lmax, nl = 1000, 10000, 100
lbd = np.linspace(lmin, lmax, nl)
lbd *= 1e-8  # angstrom to cm

# stellar par
logg = 4.17 # superficial gravity
Teff = 25940 # effective temperature
M = 11.03  # stellar mass
Rs = 5.11  # equatorial radius
L = 8200.  # luminosity
oblat = 5.11 / 4.5 #oblatness

Rp = Rs / oblat
Td = .6 * Teff
H0 = (k * Td / mi / mh * (Rs * Rsol)**3 / G / (M * Msol))**.5

# opacity
def lbdn(n):
	return n**2 / R / Z
def Xn(n, Td):
	return 2 * np.pi**2 * me * el**4 / (n**2 * h**2 * k * Td)
def N01(N0e, Td):
	return h**3 / (2 * np.pi * me * k * Td)**1.5 * N0e**2 * np.exp(Xn(1, Td))
def N1(Ne, Td):
	return h**3 / (2 * np.pi * me * k * Td)**1.5 * Ne**2 * np.exp(Xn(1, Td))
def kappa_mcdavid(lbd, Td, eta=7):
	#l = 1e-4 * lbd
	C0 = 32 / 3 / np.sqrt(3) * np.pi**2 * el**6 * R * Z**4 * np.exp(-Xn(1, Td))\
	/ mh / c**3 / h**3 * lbd**3 * (1 - np.exp(-h * c / lbd / k / Td))
	Cbf = 0
	for i in np.arange(1, eta+1, 1):
		tmp = np.exp(Xn(i, Td)) / i**3 #* np.ones(len(lbd))
		#tmp[lbd>lbdn(i)] = 0
		if lbd > lbdn(i):
			tmp = 0
		Cbf += tmp
	Cff = np.exp(Xn(eta+1, Td)) / 2 / Xn(1, Td)
	kap = C0 * (Cbf + Cff)
	return kap
def kv_md(lbd, Td, r, Ne):
	kmd = kappa_mcdavid(lbd, Td, eta=7) * mh * N1(Ne, Td)
	return kmd

# radial optical depth for electron scattering
def taue(Rs, r, Ne):
    sige = 8 * np.pi / 3 * ((el ** 2) / (me * c ** 2)) ** 2
    return np.trapz(Ne * sige, x=r) * Rs * Rsol

# net polarization
def p0(Rs, r, Ne, theta, m):
    return 3 / 16 * ((m+1.5) - 1) * sp.special.beta(((m+1.5) - 1) / 2, 3 / 2) * taue(Rs, r, Ne) * np.sin(theta*np.pi/180.) * np.cos(theta*np.pi/180.) ** 2 * np.sin(i*np.pi/180.) ** 2

# black body
def B(lbd, T):
	return (2 * h * c ** 2) / (lbd ** 5) * 1 / (np.exp(h * c / (lbd * k * T)) - 1)

# stellar flux
def Fs(lbd, Teff, logg):
	l, f, _ = spt.kuruczflux(Teff, logg)
	l *= 1e1  # Ang
	f = 2.99792458E+18 * f * l ** -2 * 4 * np.pi  # erg/s/cm2/A
	return 1e8 * np.interp(lbd, l * 1e-8, f)  # flux at surface, cgs

# radial optical depth for neutral hydrogen absorption
def taua(lbd, Td, Rs, r, Ne):
	t = Rs * Rsol * np.trapz(kv_md(lbd, Td, r, Ne), x=r)
	return t

# envelope luminosity
def L(lbd, Td, Rs, r, Ne, theta):
	j = kv_md(lbd, Td, r, Ne) * B(lbd, Td)
	return (Rs * Rsol) ** 3 * np.sin(theta*np.pi/180.) * 16 * np.pi ** 2 * np.trapz(j * r ** 2, x=r)

# polarization
def p(lbd, Td, Teff, Rs, r, Ne, theta, m):
	plbd = p0(Rs, r, Ne, theta, m) * np.exp(-taua(lbd, Td, Rs, r, Ne)) * 1 / (
				1 + L(lbd, Td, Rs, r, Ne, theta) / (4 * np.pi * (Rs * Rsol) ** 2 * Fs(lbd, Teff, logg)))
	return plbd * 100

def pol(alpha, sigma0, theta, m, lbdc):
    '''Return the polarization fraction (%) at a certain wavelength

    alpha: disk viscosity parameter
    sigma0: surface density at the base of the disk in g/cm**2
    theta: disk opening angle in degrees
    m: surface density power law exponent
    lbdc: central walvelength in cm

    Any arbitrary surface density profile (Sigma(r)) can be used instead of this one,
    including 2D arrays with multiple density profiles arrays (e.g. in different time steps)
    '''
    
	r = np.linspace(1, 115, 399)
	Sigma = sigma0 * r**(-m)

	# electron density
	Ne = Sigma / (2 * mi * mh * np.tan(theta*np.pi/180.) * r * Rs * Rsol)

	pol = p(lbdc, Td, Teff, Rs, r, Ne, theta, m)

	return pol

lbdcV = .545e-4
fwhmV = .84e-5

