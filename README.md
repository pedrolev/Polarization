Be Star Polarization Model
Python implementation of the polarization model for Be star circumstellar disks based on the approximations developed in McDavid, D. 2001, ApJ, 553, 1027 and Bjorkman, J.E. & Bjorkman, K.S. 1994, ApJ, 436, 818

Overview
This code calculates the continuum polarization percentage produced by electron scattering in an axisymmetric disk around a rapidly rotating Be star. 
The model implements McDavid's computationally efficient approximation for hydrogen bound-free and free-free opacity, providing a practical alternative to detailed radiative transfer calculations while maintaining physical accuracy.

Three Opacity Methods can be used:
McDavid LTE - Original LTE approximation from McDavid (2001)
Bjorkman - Incorporates specific treatments for n=2 and n=3 hydrogen levels
McDavid NLTE - LTE approximation with NLTE corrections using departure coefficients

Dependencies: NumPy, SciPy, Matplotlib, pyhdust

Model Assumptions
Axisymmetric disk - No azimuthal structure
Geometrically thin - Simplified vertical structure
Isothermal disk - Constant temperature with radius
Single scattering - Multiple scattering effects neglected

The code supports custom density profiles.

Output
The pol() function returns polarization as a percentage. For array inputs (multiple density profiles), it returns an array of polarization values.
