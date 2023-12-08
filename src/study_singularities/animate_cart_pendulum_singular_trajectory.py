from systems.cart_pendulum.cas.dynamics import (
  get_cart_pend_dynamics,
  CartPendParameters
)
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
from motion_planner.trajectory import MechanicalSystemTrajectory
from systems.cart_pendulum.vis.vis import animate
from simple_planner_demo.servo_connections_planner import (
  get_reduced_dynamics,
  ServoConnection,
  get_original_system_trajectory,
  ReducedDynamics,
)
from simple_planner_demo.singular_reduced_dynamics import (
  get_siungular_trajectory
)


def show_trajectory(traj : MechanicalSystemTrajectory):
  fig, axes = plt.subplots(2, 2, sharex=True, num='phase trajectory components')
  plt.sca(axes[0,0])
  plt.grid(True)
  plt.plot(traj.q[:,1], traj.q[:,0])
  plt.ylabel(R'$x$')
  plt.sca(axes[0,1])
  plt.grid(True)
  plt.plot(traj.q[:,1], traj.dq[:,1])
  plt.ylabel(R'$\dot\phi$')
  plt.sca(axes[1,0])
  plt.grid(True)
  plt.xlabel(R'$\phi$')
  plt.plot(traj.q[:,1], traj.dq[:,0])
  plt.ylabel(R'$\dot x$')
  plt.sca(axes[1,1])
  plt.grid(True)
  plt.plot(traj.q[:,1], traj.u)
  plt.ylabel(R'$u$')
  plt.xlabel(R'$\phi$')
  plt.tight_layout()
  return fig

def plot_reduced_coefficients(reduced : ReducedDynamics):
  s1,s2 = reduced.diap
  s = np.linspace(s1, s2, 600)
  alpha = np.zeros(len(s), float)
  beta = np.zeros(len(s), float)
  gamma = np.zeros(len(s), float)
  for i in range(len(s)):
    alpha[i] = float(reduced.alpha(s[i]))
    beta[i] = float(reduced.beta(s[i]))
    gamma[i] = float(reduced.gamma(s[i]))

  ax = plt.subplot(311)
  plt.grid(True)
  plt.axhline(0, ls='--', color='grey')
  plt.ylabel(R'$\alpha$')
  plt.plot(s, alpha)

  plt.subplot(312, sharex=ax)
  plt.grid(True)
  plt.axhline(0, ls='--', color='grey')
  plt.ylabel(R'$\beta$')
  plt.plot(s, beta)

  plt.subplot(313, sharex=ax)
  plt.grid(True)
  plt.axhline(0, ls='--', color='grey')
  plt.ylabel(R'$\gamma$')
  plt.plot(s, gamma)

def anim_cart_pend_singular_traj():
  par = CartPendParameters(
    link_lengths=[1.],
    mass_center=[1.],
    masses=[0.1, 0.1],
    gravity_accel=1
  )
  mechsys = get_cart_pend_dynamics(par)
  theta = ca.SX.sym('arg')
  q_singular = ca.vertcat(0, 2.0)
  c1 = 0.6
  v1 = c1 * ca.substitute(ca.pinv(mechsys.M) @ mechsys.B, mechsys.q, q_singular)
  v2 = ca.DM([-1., 0.5])
  Q_expr = q_singular + v1 * theta + v2 * theta**2

  Q_fun = ca.Function('Q', [theta], [Q_expr])
  connection = ServoConnection(fun=Q_fun, diap=[-0.5, 0.5])
  reduced = get_reduced_dynamics(connection, mechsys)
  rtraj = get_siungular_trajectory(reduced, -0.3, 0.3, 0.)
  traj = get_original_system_trajectory(rtraj, connection, mechsys)

  fig = show_trajectory(traj)
  fig.savefig('fig/cart_pend_horiz_traj.png', transparent=True)

  plt.figure('anim')
  anim = animate(traj.t, traj.q, par)
  anim.save('fig/cart_pend_horiz_traj.gif')


if __name__ == '__main__':
  plt.rcParams.update({
      "text.usetex": True,
      "font.family": "Helvetica",
      'font.size': 16
  })
  anim_cart_pend_singular_traj()
