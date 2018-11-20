# group 3 Louis, Nusrat, Henil - Project 4
import numpy as np
def create_laplacian_2d(n_boxes, box_length, pbc=True):
	if n_boxes < 2:
		raise ValueError('We need at least two grid points')                    
	if box_length <= 0.0:                                                       
		raise ValueError('We need a positive length')                           
	if pbc not in (True, False):                                                
		raise TypeError('We need a boolean as pbc')                             
	laplacian = np.zeros((n_boxes,n_boxes))
	m = (n_boxes / box_length) ** 2
	for i in range(n_boxes):
		laplacian[i,i] -= 4 * m
		laplacian[i, (i+1) % n_boxes] = 1 * m
		laplacian[(i+1) % n_boxes, i] = 1 * m
		laplacian[i, (i+2) % n_boxes] = 1  * m
		laplacian[(i+2) % n_boxes, i] = 1  * m
	if not pbc:
		laplacian[0, -1] = 0
		laplacian[-1, 0] = 0
		laplacian[0, -2] = 0
		laplacian[1, -1] = 0
		laplacian[-1, 1] = 0
		laplacian[-2, 0] = 0
	return laplacian
