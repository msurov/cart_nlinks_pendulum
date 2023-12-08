from sym_dynamics.cart_pend_dynamics import get_cart_pend_dynamics, CartPendParameters
from sym_dynamics.mechsys import get_mechsys_normal_form, get_full_energy
from common.simulation import simulate
from systems.cart_pendulum.vis.vis import animate
import numpy as np
import matplotlib.pyplot as plt
import sympy as sy
from common.perf_counter import PerfCounter


def test():
  pc = PerfCounter()
  par = CartPendParameters(
    link_lengths = [1.0] * 3,
    mass_center = [0.4] * 3,
    masses = [0.1] * 4,
    gravity_accel = 9.81
  )
  pc('get_cart_pend_dynamics')
  dynamics = get_cart_pend_dynamics(par, simplify=True)
  pc('get_mechsys_normal_form')
  sys = get_mechsys_normal_form(dynamics)
  dim = par.nlinks + 1

  input = lambda _,__: [0.]
  np.random.seed(0)
  x0 = np.array([
    1., 2., 3., 4., 5., 6., 7., 8.
  ])

  pc('simulate')
  simres = simulate(sys, input, [0., 15.], x0, 1e-2, max_step=1e-3)

  pc('get_full_energy')
  args = sy.Tuple(*dynamics.q, *dynamics.dq)
  full_energy_expr = get_full_energy(dynamics)
  full_energy_expr = sy.simplify(full_energy_expr)
  full_energy = sy.lambdify(args, full_energy_expr, 'numpy')
  energy = [full_energy(*x) for x in simres.x]

  pc('plot')
  fig, axes = plt.subplots(1, 3, figsize=(10, 5))

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
  test()
