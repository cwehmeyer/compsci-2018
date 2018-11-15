import numpy as np


def langevin(
        force, n_steps, x_init, v_init, mass,
        time_step=0.001, damping=0.1, beta=1.0):
    """Langevin integrator for initial value problems

    This function implements the BAOAB algorithm of Benedict Leimkuhler
    and Charles Matthews. See J. Chem. Phys. 138, 174102 (2013) for
    further details.

    Arguments:
        force (function): computes the forces of a single configuration
        n_steps (int): number of integration steps
        x_init (numpy.ndarray(n, d)): initial configuration
        v_init (numpy.ndarray(n, d)): initial velocities
        mass (numpy.ndarray(n)): particle masses
        time_step (float): time step for the integration
        damping (float): damping term, use zero if not coupled
        beta (float): inverse temperature

    Returns:
        x (numpy.ndarray(n_steps + 1, n, d)): configuraiton trajectory
        v (numpy.ndarray(n_steps + 1, n, d)): velocity trajectory
    """
    shape = list(x_init.shape)
    th = 0.5 * time_step
    thm = 0.5 * time_step / mass[:, None]
    edt = np.exp(-damping * time_step)
    sqf = np.sqrt((1.0 - edt ** 2) / (beta * mass))[:, None]
    x = np.zeros([n_steps + 1] + shape)
    v = np.zeros_like(x)
    x[0, :, :] = x_init
    v[0, :, :] = v_init
    f = force(x[0])
    for i in range(n_steps):
        v[i + 1, :, :] = v[i] + thm * f
        x[i + 1, :, :] = x[i] + th * v[i + 1]
        v[i + 1, :, :] = edt * v[i + 1] + sqf * np.random.randn(*shape)
        x[i + 1, :, :] = x[i + 1] + th * v[i + 1]
        f[:, :] = force(x[i + 1])
        v[i + 1, :, :] = v[i + 1] + thm * f
    return x, v
