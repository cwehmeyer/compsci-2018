import numpy as np

def jacobi(charge_dist, boxsize, maxiter=1000, tol=10**(-5), epsilon0=1.0):
     
    """An implementation of the Jacobi-method for solving the periodic discretized Poisson-equation in n dimensions.
    Returns the potential as a n-dimensional array.

    Arguments:
        charge_dist: n-dimensional array consisting of entries of type float that define the charge distribution.
        boxsize: tuple of length n, which containts the lengths of the box-axes.
        maxiter: int value that sets the maximum number of iterations being performed by the algorithm.
        tol: float that sets the break-off-condition for the algorithm.
        epsilon0: float, scaling factor of the charge distribution.
    """
    
    if type(boxsize) in (int, float):
        boxsize = (boxsize, )
      
    h_array = np.asarray(boxsize)
    rho = np.asarray(charge_dist)
    pot = np.zeros(rho.shape)
    dim = len(h_array)
  
    if type(maxiter) is not int:
        raise TypeError("Value maxiter has to be of type int.")
    if type(tol) not in (int, float):
        raise TypeError("Value tol has to be a real numeric value.")
    if type(epsilon0) not in (int, float):
        raise TypeError("Value epsilon0 has to be a real numeric value.")
    if rho.dtype not in (int, float):
        raise TypeError("Entries of charge_dist have to be of type int or float.")
    if h_array.dtype not in (int, float):
        raise TypeError("Entries of boxsize have to be either of type int or float.")
    if h_array.ndim != 1:
        raise TypeError("Boxsize needs to be a one-dimensional array.")
    if dim != rho.ndim:
        raise ValueError("Boxsize needs to have as many values as charge_dist has axes.")
    if any(h_array==0.):
        raise ValueError("Boxsize values all have to be nonzero.") 
    if (1 or 0) in rho.shape:
        raise ValueError("All axes of charge_dist need to have more than one element.")
    if not np.isclose(np.mean(rho), 0, atol=1e-16):
        raise ValueError("The mean of charge_dist has to be zero. Consider subtracting the mean and try again.")
     
    c = 2*np.sum(1/(h_array)**2)
    dif = np.inf
    counter = 0  
    
    while tol < dif and counter < maxiter:
        pot_next = np.zeros(rho.shape)
        for i, h in enumerate(h_array[::-1]):
            pot_next += (np.roll(pot, 1, axis=i)+np.roll(pot, -1, axis=i))/h**2  
        pot_next = (1/c)*(pot_next + rho/epsilon0)
        dif = np.linalg.norm(pot_next - pot)
        pot = pot_next
        counter+= 1
        
    return pot