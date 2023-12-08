from systems.cart_pendulum.cas.dynamics import get_cart_pend_dynamics, CartPendParameters
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
from motion_planner.trajectory import MechanicalSystemTrajectory
from systems.cart_pendulum.vis.vis import animate
from simple_planner_demo.servo_connections_planner import (
  ReducedDynamics,
  get_reduced_dynamics,
  get_reduced_periodic_solution,
  ServoConnection,
  get_original_system_trajectory
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

def demo():
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

  reduced_traj = get_reduced_periodic_solution(reduced, [0.5, 0.], period_max=10)
  traj = get_original_system_trajectory(reduced_traj, connection, mechsys)

  plt.figure('draw')
  fig = show_trajectory(traj)
  fig.savefig('fig/one_link_pendulum_upper_oscllations.png', transparent=True)

  plt.figure('anim')
  anim = animate(traj.t, traj.q, par)
  anim.save('fig/one_link_pendulum_upper_oscllations.gif')

if __name__ == '__main__':
  plt.rcParams.update({
      "text.usetex": True,
      "font.family": "Helvetica",
      'font.size': 16
  })
  demo()
