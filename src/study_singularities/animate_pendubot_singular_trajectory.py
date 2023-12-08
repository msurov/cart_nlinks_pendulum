from systems.double_pendulum.cas.dynamics import (
  get_double_pendulum_dynamics,
  DoublePendulumParameters
)
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
from motion_planner.trajectory import MechanicalSystemTrajectory
from systems.double_pendulum.vis.vis import animate
from simple_planner_demo.servo_connections_planner import (
  get_reduced_dynamics,
  ServoConnection,
  get_original_system_trajectory
)
from simple_planner_demo.singular_reduced_dynamics import (
  get_siungular_trajectory,
  plot_reduced_coefficients
)

def show_trajectory(traj : MechanicalSystemTrajectory):
  fig, axes = plt.subplots(2, 2, sharex=True, num='phase trajectory components')
  plt.sca(axes[0,0])
  plt.grid(True)
  plt.plot(traj.q[:,1], traj.q[:,0])
  plt.ylabel(R'$\theta_1$')
  plt.sca(axes[0,1])
  plt.grid(True)
  plt.plot(traj.q[:,1], traj.dq[:,1])
  plt.ylabel(R'$\dot\theta_2$')
  plt.sca(axes[1,0])
  plt.grid(True)
  plt.xlabel(R'$\theta_2$')
  plt.plot(traj.q[:,1], traj.dq[:,0])
  plt.ylabel(R'$\dot \theta_1$')
  plt.sca(axes[1,1])
  plt.grid(True)
  plt.plot(traj.q[:,1], traj.u)
  plt.ylabel(R'$u$')
  plt.xlabel(R'$\theta_2$')
  plt.tight_layout()
  return fig

def anim_pendubot_singular_traj():
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
  theta = ca.SX.sym('arg')
  q_singular = ca.vertcat(1.0, 0.4)
  v1 = 1.0 * ca.substitute(ca.pinv(mechsys.M) @ mechsys.B, mechsys.q, q_singular)
  v2 = ca.DM([1.7, 0.])
  Q_expr = q_singular + v1 * theta + v2 * theta**2

  Q_fun = ca.Function('Q', [theta], [Q_expr])
  connection = ServoConnection(fun=Q_fun, diap=[-0.1, 0.1])
  reduced = get_reduced_dynamics(connection, mechsys)

  print('dalpha', reduced.dalpha(0.))
  print('-beta/alpha\'', -reduced.beta(0.) / reduced.dalpha(0.))
  print('-gamma/beta', -reduced.gamma(0.) / reduced.beta(0.))

  # plot_reduced_coefficients(reduced)
  # plt.show()
  # exit()

  rtraj = get_siungular_trajectory(reduced, -0.05, 0.1, 0.)
  traj = get_original_system_trajectory(rtraj, connection, mechsys)

  fig = show_trajectory(traj)
  fig.savefig('fig/pendubot_horiz_traj.png', transparent=True)

  plt.figure('anim')
  anim = animate(traj.t, traj.q, par)
  anim.save('fig/pendubot_horiz_traj.gif')

if __name__ == '__main__':
  plt.rcParams.update({
      "text.usetex": True,
      "font.family": "Helvetica",
      'font.size': 16
  })
  anim_pendubot_singular_traj()
