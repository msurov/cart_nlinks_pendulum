from sym_dynamics.cart_pend_dynamics import CartPendParameters
from motion_planner.servo_connection_planner import (
  find_sample_trajectory,
  MechanicalSystemPeriodicTrajectory
)
import matplotlib.pyplot as plt
from systems.cart_pendulum.vis.vis import animate
import sympy as sy
import numpy as np


def duplicate_traj(traj : MechanicalSystemPeriodicTrajectory) -> MechanicalSystemPeriodicTrajectory:
  return MechanicalSystemPeriodicTrajectory(
    t = np.concatenate((traj.t, traj.t[-1] + traj.t[1:]), axis=0),
    q = np.concatenate((traj.q, traj.q[1:]), axis=0),
    dq = np.concatenate((traj.dq, traj.dq[1:]), axis=0),
    ddq = np.concatenate((traj.ddq, traj.ddq[1:]), axis=0),
    u = np.concatenate((traj.u, traj.u[1:]), axis=0),
    period = traj.period,
    periodicity_deviation = traj.periodicity_deviation
  )

def test():
  par = CartPendParameters(
    link_lengths = [sy.sympify(1), sy.sympify(1)],
    mass_center = [sy.sympify(1)/2, sy.sympify(1)/2],
    masses = [sy.sympify(1) / 8, sy.sympify(1) / 8, sy.sympify(1) / 8],
    gravity_accel = sy.sympify(10)
  )
  traj = find_sample_trajectory(par)

  fig, axes = plt.subplots(1, 3, figsize=(16, 5))
  plt.sca(axes[0])
  plt.grid(True)
  plt.plot(traj.q[:,0], traj.dq[:,0], label=R'$\dot x$')
  plt.plot(traj.q[:,0], traj.dq[:,1], label=R'$\dot \theta_1$')
  plt.plot(traj.q[:,0], traj.dq[:,2], label=R'$\dot \theta_2$')
  plt.xlabel(R'$x$')
  plt.legend()

  plt.sca(axes[1])
  plt.grid(True)
  plt.plot(traj.q[:,0], traj.q[:,1], label=R'$\theta_1$')
  plt.plot(traj.q[:,0], traj.q[:,2], label=R'$\theta_2$')
  plt.xlabel(R'$x$')
  plt.legend()

  plt.sca(axes[2])
  plt.grid(True)
  traj2 = duplicate_traj(traj)
  anim = animate(traj2.t, traj2.q, par)

  plt.tight_layout()
  anim.save('fig/periodic_motion_anim.gif')


if __name__ == '__main__':
  test()
