from cas_dynamics.dynamics import get_cart_pend_dynamics, CartPendParameters
from cas_dynamics.mechsys import get_mechsys_normal_form, get_full_energy
from common.simulation import simulate
from vis.anim import animate
import numpy as np
import matplotlib.pyplot as plt


def demo():
  par = CartPendParameters(
    link_lengths = [1.0] * 3,
    mass_center = [0.4] * 3,
    masses = [0.1] * 4,
    gravity_accel = 9.81
  )
  mechsys = get_cart_pend_dynamics(par)
  sys = get_mechsys_normal_form(mechsys)
  dim = par.nlinks + 1

  input = lambda _,__: [0.]
  x0 = np.array([
    1., 2., 3., 4., 5., 6., 7., 8.
  ])

  simres = simulate(sys, input, [0., 15.], x0, 1e-2, max_step=1e-3)

  fig, axes = plt.subplots(1, 3, figsize=(10, 5))
  energy_fun = get_full_energy(mechsys)
  energy = [float(energy_fun(x)) for x in simres.x]

  plt.sca(axes[0])
  plt.grid(True)
  plt.plot(simres.t, energy)
  plt.legend(['full energy'])

  plt.sca(axes[1])
  plt.grid(True)
  plt.plot(simres.t, simres.x[:,0:dim])
  plt.legend(['x', R'$\theta_1$', R'$\theta_2$'])

  plt.sca(axes[2])
  anim = animate(simres.t, simres.x[:,0:dim], par)
  anim.save('fig/free_motion_anim.gif')

if __name__ == '__main__':
  demo()
