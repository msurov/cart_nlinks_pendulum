from motion_planner.servo_connection_planner import find_sample_trajectory
import matplotlib.pyplot as plt
from systems.cart_pendulum.sym.dynamics import CartPendParameters
import sympy as sy


def test():
  par = CartPendParameters(
    link_lengths = [sy.sympify(1), sy.sympify(1)],
    mass_center = [sy.sympify(1)/2, sy.sympify(1)/2],
    masses = [sy.sympify(1) / 8, sy.sympify(1) / 8, sy.sympify(1) / 8],
    gravity_accel = sy.sympify(10)
  )
  traj = find_sample_trajectory(par)

  fig, axes = plt.subplots(2, 2)
  plt.sca(axes[0,0])
  plt.grid(True)
  plt.plot(traj.t, traj.q)
  plt.xlabel('$t$')
  plt.legend([R'$x$', R'$\theta_1$', R'$\theta_2$'])

  plt.sca(axes[0,1])
  plt.grid(True)
  plt.plot(traj.q[:,2], traj.q[:,0], label=R'$x$')
  plt.plot(traj.q[:,2], traj.q[:,1], label=R'$\theta_1$')
  plt.xlabel(R'$\theta_2$')

  plt.sca(axes[1,0])
  plt.grid(True)
  plt.plot(traj.t, traj.dq)
  plt.legend([R'$\dot x$', R'$\dot\theta_1$', R'$\dot\theta_2$'])
  plt.xlabel('$t$')

  plt.sca(axes[1,1])
  plt.grid(True)
  plt.plot(traj.t, traj.u)
  plt.legend(['$u$'])
  plt.xlabel(R'$t$')

  plt.tight_layout()
  plt.show()

if __name__ == '__main__':
  test()
