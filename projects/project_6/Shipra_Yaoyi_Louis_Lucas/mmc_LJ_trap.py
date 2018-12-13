import numpy as np
##################### declare constants ########################
N_particles= 4
mc_steps = 1000
particle_radius = 1
potential_depth = 1
spring_constant = 1
beta = 1
dimension = 2
inital_density = 0.5
initial_length = (N_particles/inital_density)**(1/dimension)
scaling = 0.1                                       
##################### distances #################################
tril_vector = np.tril_indices(N_particles,-1)
def get_distances(coordinates):
    dist_matrix = np.linalg.norm(
        coordinates[:, None, :] - coordinates[None, :, :],
        axis=-1)             
    return dist_matrix[tril_vector]
################### Lennard-Jones ###############################
def LJ_potential(coordinates): 
	u_LJ = 0
	d = get_distances(coordinates)
	for _ in range(len(d)):
		if d[_] == 0:
			u_LJ = 1e500
			print('divided by zero')
		elif d[_] < 2.5 * particle_radius:
			attractive_part_of_potential = (particle_radius/d[_])**(6.)
			u_LJ += (attractive_part_of_potential) ** 2. - attractive_part_of_potential
	return u_LJ * 4 * potential_depth
################### harmonic_Potential ###########################	
def harmonic_Potential(coordinates):
	u_harmonic = 0
	for _ in range(N_particles):
		u_harmonic += spring_constant * 0.5 * np.linalg.norm(coordinates[_])**2. 
	return u_harmonic
################## Metropolis Monte Carlo ########################
def mmc():
	coordinates = np.zeros([mc_steps, N_particles, dimension])
	u = np.zeros(mc_steps)
	initial_coordinates = np.random.rand(N_particles, dimension) * initial_length - 0.5 * initial_length * np.ones([N_particles, dimension])
	coordinates[0] = initial_coordinates
	for t in range(1,mc_steps):
		choose_particle = np.random.randint(0, high=N_particles, size=None, dtype='l')
		random_vector = np.zeros([N_particles,dimension])	
		random_vector[choose_particle] = np.random.rand(dimension)-0.5
		coordinates_ = coordinates[t-1] + scaling * random_vector / np.linalg.norm(random_vector)
		u_  = LJ_potential(coordinates_)     + harmonic_Potential(coordinates_)
		u[t-1] = LJ_potential(coordinates[t-1])	   + harmonic_Potential(coordinates[t-1])
		if u_ <= u[t-1] \
		or np.random.rand() < np.exp(beta*(u[t-1]-u_)):
			coordinates[t] = coordinates_
			u[t] = u_  
		else:
			coordinates[t] = coordinates[t-1]
			u[t] = u[t-1]
	return u, coordinates
