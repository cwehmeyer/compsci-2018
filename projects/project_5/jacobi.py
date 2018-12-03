import numpy as np

def jacobi(charge_dist, boxsize, maxiter=100, tol=10**(-5), epsilon0=1.0):
    
    if type(boxsize) in (int, float):
        boxsize = (boxsize, )
      
    h_array = np.array(boxsize)
    rho = np.asarray(charge_dist)
    pot = np.zeros(rho.shape)
    dim = len(h_array)
  
    if dim != rho.ndim:
        raise ValueError("Boxsize needs to have as many values a charge_dist has axes.")
    if any(h_array==0.):
        raise ValueError("Boxsize values all have to be nonzero.") 
    if rho.dtype not in (int, float):
        raise TypeError("Entries of charge_dist have to be of type int or float.")
    if h_array.dtype not in (int, float):
        raise TypeError("Entries of boxsize have to be either of type int or float.")
     
    c = 2*np.sum(np.array([1/h**2 for h in h_array]))
    dif = np.inf
    counter = 0  
    
    while tol < dif and counter < maxiter:
        pot_next = np.zeros(rho.shape)
        for i, h in enumerate(h_array[::-1]):
            pot_next += (np.roll(pot, 1, axis=i)+np.roll(pot, -1, axis=i))/h**2  
        pot_next = (1/c)*(pot_next + rho/epsilon0)
        dif = np.linalg.norm(pot_next - pot)
        pot = pot_next[::]
        counter+= 1
        
    return pot