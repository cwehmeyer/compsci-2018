import numpy as np

# temporarily define all constants here
epsilon_lj = 1.
sigma_lj = 1.
cutoff_lj = 3. * sigma_lj

def LJ_potential(q):
    """Calculate the system Lennard-Jones potential based on given particle coordinates.
    
    Input:
        q (iterable of (#dimension,)-arrays): list or array of particle coordinates.
    
    Output:
        pot (float): sum of all pairwise LJ potentials.    
    """
    
    # TODO: input check
    
    n_particle = np.shape(q)[0]
    pot = 0.
    for i in range(n_particle - 1):
        for j in range(1, n_particle):
            pot += _LJ_potential_pair(q[i], q[j])
    return pot

def _LJ_potential_pair(qi, qj):
    """Calculate pairwise Lennard-Jones potential.
        (require global epsilon_lj, sigma_lj, cutoff_lj)
    """
    rij = np.linalg.norm(qi - qj)
    if rij <= 0 or rij > cutoff_lj:
        return 0.
    return 4 * epsilon_lj * (sigma_lj ** 12 / rij ** 12 - sigma_lj ** 6 / rij ** 6)

def LJ_force(q):
    """Calculate the system Lennard-Jones forces based on given particle coordinates.
    
    Input:
        q (iterable of (#dimension,)-arrays): list or array of particle coordinates.
        k (int): index of the particle for force calc.
    
    Output:
        force ((n, #dimension)-array): system Lennard-Jones forces on particle *0~(n-1)*.    
    """
    
    # TODO: input check
    
    n_particle = np.shape(q)[0]
    ndim = np.shape(q)[1]
    force = np.zeros((n_particle, ndim))
    for i in range(n_particle):
        for j in range(n_particle):
            # pairwise force calc will return 0 for identical two particles.
            force[i] += _LJ_force_pair(q[i], q[j])
    return force

def _LJ_force_pair(qk, qj):
    """Calculate pairwise Lennard-Jones force on paricle *k*.
        (require global epsilon_lj, sigma_lj, cutoff_lj)
    """
    rjk = np.linalg.norm(qj - qk)
    ndim = np.shape(qk)[0]
    if rjk <= 0 or rjk > cutoff_lj:
        return np.zeros(ndim)
    # return 4 * epsilon_lj * (12 * sigma_lj ** 12 / rjk ** 14 - 6 * sigma_lj ** 6 / rjk ** 8) * (qj - qk)
    return 4 * epsilon_lj * (12 * sigma_lj ** 12 / rjk ** 14 - 6 * sigma_lj ** 6 / rjk ** 8) * (qk - qj)
