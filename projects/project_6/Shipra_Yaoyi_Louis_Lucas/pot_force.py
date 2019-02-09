import numpy as np
from LJ import LJ_force, LJ_potential

# temporarily define all constants here
k_harmonic = 1

def harmonic_potential(q):
    """Calculate a well-like harmonic potential based on given particle coordinates.
    Centered at the origin.
    
    Input:
        q (iterable of (#dimension,)-arrays): list or array of particle coordinates.
    
    Output:
        pot (float): sum of all harmonic potentials.    
    """
    
    # TODO: input check
    
    return k_harmonic * np.square(q).sum()

def harmonic_force(q):
    """Calculate a well-like harmonic force based on given particle coordinates.
    
    Input:
        q (iterable of (#dimension,)-arrays): list or array of particle coordinates.
    
    Output:
        force (float): force from the harmonic well on each particle.    
    """
    
    return -2 * k_harmonic * q

def all_potential(q):
    return LJ_potential(q) + harmonic_potential(q)

def all_force(q):
    return LJ_force(q) + harmonic_force(q)
