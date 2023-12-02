from cas_dynamics.dynamics import get_cart_pend_dynamics, CartPendParameters
from cas_dynamics.mechsys import get_mechsys_normal_form
from common.simulation import simulate
import numpy as np
import matplotlib.pyplot as plt


def demo():
  par = CartPendParameters(
    link_lengths=[1, 1],
    mass_center=[0.5, 0.5],
    masses=[0.1, 0.1, 0.1],
    gravity_accel=9.81
  )
  mechsys = get_cart_pend_dynamics(par)
  rhs = get_mechsys_normal_form(mechsys)
  q0 = np.zeros(mechsys.qdim)
  dq0 = 1e-3 * np.random.normal(size=mechsys.qdim)
  st0 = np.concatenate((q0, dq0))
  fb = lambda _, x: np.array([0.])
  simres = simulate(rhs, fb, [0, 10], st0, 1e-2)

  plt.plot(simres.t, simres.x)
  plt.show()


if __name__ == '__main__':
  demo()
