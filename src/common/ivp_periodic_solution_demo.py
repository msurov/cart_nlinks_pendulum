import matplotlib.pyplot as plt
import numpy as np
from common.integrate import solve_periodic_ivp
from scipy.interpolate import make_interp_spline


def sample():

  def sys(_, st):
    x, dx = st
    return np.array([dx, -np.sin(x) ])

  tstart = 10.
  xstart = [2., 0.]
  step = 0.01
  sol = solve_periodic_ivp(sys, tstart, xstart, step, None, 
                           eps=1e-3, max_step=1e-4, atol=1e-12, rtol=1e-12, nsteps=100)
  sp = make_interp_spline(sol.t, sol.x, k = 3, bc_type='periodic')

  plt.subplot(221)
  plt.title('phase trajectory')
  plt.grid(True)
  plt.plot(sol.x[:,0], sol.x[:,1])

  plt.subplot(222)
  plt.title('deviation')
  plt.grid(True)
  plt.plot(sol.t, (sp(sol.t) + sp(2*tstart - sol.t))[:,1], label='speed err')
  plt.plot(sol.t, (sp(sol.t) - sp(2*tstart - sol.t))[:,0], label='angle err')
  plt.legend()

  plt.subplot(223)
  plt.title('solution components')
  plt.grid(True)
  plt.plot(sol.t, sol.x[:,0], color='b')
  plt.plot(sol.t + sol.period, sol.x[:,0], color='b', label='angle')
  plt.plot(sol.t, sol.x[:,1], color='g')
  plt.plot(sol.t + sol.period, sol.x[:,1], color='g', label='speed')
  plt.axvline(sol.t[0], color='grey')
  plt.axvline(sol.t[0] + sol.period, color='grey')
  plt.axvline(sol.t[0] + 2 * sol.period, color='grey')
  plt.legend()

  plt.subplot(224)
  plt.title('full energy')
  plt.grid(True)
  energy = sol.x[:,1]**2 / 2 - np.cos(sol.x[:,0])
  plt.plot(sol.t, energy)
  plt.plot(sol.t + sol.period, energy)
  plt.axvline(sol.t[0], color='grey')
  plt.axvline(sol.t[0] + sol.period, color='grey')
  plt.axvline(sol.t[0] + 2 * sol.period, color='grey')

  plt.tight_layout()
  plt.show()

if __name__ == '__main__':
  sample()
