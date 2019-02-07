################### create Movie of configs #####################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mmc_LJ_trap import *
matplotlib.use("Agg")
u, coordinates = mmc()
import matplotlib.animation as manimation

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=20, metadata=metadata)

fig = plt.figure()
l, = plt.plot([], [], 'ro')
plt.setp(l, markersize=30)
plt.setp(l, markerfacecolor='C0') 


plt.xlim(-5, 5)
plt.ylim(-5, 5)

x, y = np.zeros(N_particles), np.zeros(N_particles)

with writer.saving(fig, "movie_of_configs.mp4", 100):
	for _ in range(mc_steps):
		coord = coordinates[_]
		for i in range(N_particles):
			x[i] = coord[i, -1]
			y[i] = coord[i, 0]
		l.set_data(x, y)
		writer.grab_frame()
