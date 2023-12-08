from ..cas.dynamics import get_double_pendulum_dynamics, DoublePendulumParameters
from dynamics.cas_mechsys import get_mechsys_normal_form, get_full_energy
from common.simulation import simulate
from systems.double_pendulum.vis import animate
import numpy as np
import matplotlib.pyplot as plt


def animate_free_motion():
  par = DoublePendulumParameters(
    link_length_1=1,
    link_length_2=1,
    mass_center_1=0.5,
    mass_center_2=0.5,
    mass_1=1,
    mass_2=1,
    gravity_accel=1,
    actuated_joint=0
  )
  mechsys = get_double_pendulum_dynamics(par)
  sys = get_mechsys_normal_form(mechsys)

  input = lambda _,__: [0.]
  x0 = np.array([
    0., 0., 1e-3, 1e-5
  ])

  simres = simulate(sys, input, [0., 10.], x0, 1e-2, max_step=1e-3)

  fig, axes = plt.subplots(1, 3, figsize=(10, 5))
  energy_fun = get_full_energy(mechsys)
  energy = [float(energy_fun(x)) for x in simres.x]

  plt.sca(axes[0])
  plt.grid(True)
  plt.plot(simres.t, energy)
  plt.legend(['full energy'])

  plt.sca(axes[1])
  plt.grid(True)
  plt.plot(simres.t, simres.x[:,0:mechsys.qdim])
  plt.legend([R'$\theta_1$', R'$\theta_2$'])

  plt.sca(axes[2])
  anim = animate(simres.t, simres.x[:,0:mechsys.qdim], par)
  anim.save('fig/pendubot_free_motion_anim.gif')
