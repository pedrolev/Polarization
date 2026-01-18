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
mu = 0.6
Z = 1
G = 6.67430e-8

lbd2 = (Z**2 * R * 1/4)**-1
lbd3 = (Z**2 * R * 1/9)**-1


# opacity from mcdavid
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

# opacity from mcdavid with NLTE correction
def qlte(n, Td, Rs, r, Ne):
	cima = n**2 * np.exp(Xn(n, Td)-Xn(1, Td)) * np.trapz(N1(Ne, Td), x=r) * Rs * Rsol
	baixo = np.trapz(Ne**2, x=r) * Rs * Rsol
	return cima/baixo

def kappa_mcdavid_nlte(lbd, Td, Rs, r, Ne, eta=7):
	C0 = 32 / 3 / np.sqrt(3) * np.pi**2 * el**6 * R * Z**4 * np.exp(-Xn(1, Td))\
	/ mh / c**3 / h**3 * lbd**3 * (1 - np.exp(-h * c / lbd / k / Td))
	Cbf = 0
	for i in np.arange(1, eta+1, 1):
		if i == 2:
			b2 = 1.11e-8 * Td**(-2.83) / qlte(2, Td, Rs, r, Ne)
			tmp = b2 * np.exp(Xn(i, Td)) / i**3 #* np.ones(len(lbd))
			if lbd > lbdn(i):
				tmp = 0
			Cbf += tmp
		elif i == 3:
			b3 = 4.59e-13 * Td**(-2.02) / qlte(3, Td, Rs, r, Ne)
			tmp = b3 * np.exp(Xn(i, Td)) / i**3 #* np.ones(len(lbd))
			if lbd > lbdn(i):
				tmp = 0
			Cbf += tmp
		else:
			tmp = np.exp(Xn(i, Td)) / i**3 #* np.ones(len(lbd))
			if lbd > lbdn(i):
				tmp = 0
			Cbf += tmp
	Cff = np.exp(Xn(eta+1, Td)) / 2 / Xn(1, Td)
	kap = C0 * (Cbf + Cff)
	return kap

def kv_md_nlte(lbd, Td, Rs, r, Ne):
	if Ne.ndim == 2:
		kmd = np.ones((len(Ne), len(r)))
		for j in range(len(Ne)):
			k = kappa_mcdavid_nlte(lbd, Td, Rs, r, Ne[j], eta=7) * mh * N1(Ne[j], Td)
			kmd[j, :] = k
		return kmd
	else:
		kmd = kappa_mcdavid_nlte(lbd, Td, Rs, r, Ne, eta=7) * mh * N1(Ne, Td)
		return kmd

# opacity from bjorkman
def ni(i, Ne, r):
	qi = 0
	W = .5 * (1-(1-(1/r)**2)**.5)
	if i == 2:
		qi = 3.5e-21
	elif i == 3:
		qi = 4.7e-22
	return qi * Ne**2 / W

def kv_bj(lbd, Td, r, Ne):
	kb = 0
	if lbd < lbd2:
		kb = 3.692e8 * (1 - np.exp(-h * c / (lbd * k * Td))) * Td ** (-0.5) * (lbd / c) ** 3 * Ne ** 2 + (ni(2, Ne, r) * 1.4e-17 * (lbd/lbd2)**3) + (ni(3, Ne, r) * 2.2e-17 * (lbd/lbd3)**3)
	elif lbd2 < lbd < lbd3:
		kb = 3.692e8 * (1 - np.exp(-h * c / (lbd * k * Td))) * Td ** (-0.5) * (lbd / c) ** 3 * Ne ** 2 + (ni(3, Ne, r) * 2.2e-17 * (lbd/lbd3)**3)
	return kb

def kv(method, lbd, Td, Rs, r, Ne):
    if method==0:
        return kv_md(lbd, Td, r, Ne)
    elif method==1:
        return kv_bj(lbd, Td, r, Ne)
    elif method==2:
        return kv_md_nlte(lbd, Td, Rs, r, Ne)

# radial optical depth for electron scattering
def taue(Rs, r, Ne):
	sige = 8 * np.pi / 3 * ((el ** 2) / (me * c ** 2)) ** 2
	return np.trapz(Ne * sige, x=r) * Rs * Rsol

# net polarization
def p0(Rs, r, Ne, theta, m, i):
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
def taua(lbd, Td, Rs, r, Ne, method):
	t = Rs * Rsol * np.trapz(kv(method, lbd, Td, Rs, r, Ne), x=r)
	return t

# envelope luminosity
def L(lbd, Td, Rs, r, Ne, theta, method):
	j = kv(method, lbd, Td, Rs, r, Ne) * B(lbd, Td)
	return (Rs * Rsol) ** 3 * np.sin(theta*np.pi/180.) * 16 * np.pi ** 2 * np.trapz(j * r ** 2, x=r)

# polarization
def p(lbd, Teff, Td, Rs, logg, i, r, Ne, theta, m, method):
	plbd = p0(Rs, r, Ne, theta, m, i) * np.exp(-taua(lbd, Td, Rs, r, Ne, method)) * 1 / (
				1 + L(lbd, Td, Rs, r, Ne, theta, method) / (4 * np.pi * (Rs * Rsol) ** 2 * Fs(lbd, Teff, logg)))
	return plbd * 100

def pol(r, Sigma, theta, m, sp, lbdc, method):
	'''Return the polarization fraction (%) at a certain wavelength

	*alpha: disk viscosity parameter
	*sigma0: surface density at the base of the disk in g/cm**2
	r: disk radius in stellar radius
	Sigma: surface density profile
	theta: disk opening angle in degrees
	m: surface density power law exponent
	lbdc: polarization central walvelength of choice in cm
	sp: stellar parameters in cgs
	i: inclination angle of the rotation axis of the star to the line of sight

	Any arbitrary surface density profile (Sigma(r)),including 2D arrays
	with multiple density profiles arrays (e.g. in different time steps)

	r = np.linspace(1, 115, 399)
	Sigma = sigma0 * r**(-m)
	'''
	Teff, Td, Rs, logg, i = sp
	# electron density
	Ne = Sigma / (2 * mu * mh * np.tan(theta*np.pi/180.) * r * Rs * Rsol)

	pol = p(lbdc, Teff, Td, Rs, logg, i, r, Ne, theta, m, method)

	return pol
