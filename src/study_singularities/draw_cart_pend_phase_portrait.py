R"""
  Create an image of the phase portrait of the cart-pendulum upward oscillations
"""

from systems.cart_pendulum.cas.dynamics import (
  get_cart_pend_dynamics,
  CartPendParameters
)
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
from simple_planner_demo.servo_connections_planner import (
  ReducedDynamics,
  get_reduced_dynamics,
  get_reduced_periodic_solution,
  ServoConnection,
)

def show_phase_portrait(reduced : ReducedDynamics):
  x = np.arange(-0.75, 0.8, 0.1)
  y = np.arange(-1, 1, 0.1)
  nx, = x.shape
  ny, = y.shape
  vx = np.zeros((ny, nx))
  vy = np.zeros((ny, nx))

  for j in range(ny):
    for i in range(nx):
      theta = x[i]
      dtheta = y[j]
      vx[j,i] = dtheta
      ddtheta = (-reduced.beta(theta)*dtheta**2 - reduced.gamma(theta)) / reduced.alpha(theta)
      vy[j,i] = float(ddtheta)

  fig = plt.figure('phase-portrait')
  plt.axhline(0, ls='--', color='grey')
  plt.plot([0], [0], 'o', color='red')
  plt.streamplot(x, y, vx, vy)

  traj = get_reduced_periodic_solution(reduced, [-0.5, 0.], 10.)
  plt.plot(traj.q, traj.dq, '-', lw=2, color='brown')
  plt.xlabel(R'$\theta$')
  plt.ylabel(R'$\dot\theta$')
  plt.tight_layout()
  return fig

def make_phase_portrait_with_reference_phase_trajectory():
  par = CartPendParameters(
    link_lengths=[1.],
    mass_center=[1.],
    masses=[0.1, 0.1],
    gravity_accel=1
  )
  mechsys = get_cart_pend_dynamics(par)
  s = ca.SX.sym('arg')
  Q_expr = ca.vertcat(-1.6 * s, s)
  Q_fun = ca.Function('Q', [s], [Q_expr])
  connection = ServoConnection(fun=Q_fun, diap=[-np.pi, np.pi])
  reduced = get_reduced_dynamics(connection, mechsys)
  fig = show_phase_portrait(reduced)
  plt.savefig('fig/phase_portrait_reference_trajectory.png', transparent=True)

if __name__ == '__main__':
  plt.rcParams.update({
      "text.usetex": True,
      "font.family": "Helvetica",
      'font.size': 16
  })
  make_phase_portrait_with_reference_phase_trajectory()

