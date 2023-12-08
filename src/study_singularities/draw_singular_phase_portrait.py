R"""
  Create an image of the singular phase portrait of the cart-pendulum upward oscillations
"""
from systems.double_pendulum.cas.dynamics import get_double_pendulum_dynamics, DoublePendulumParameters
from systems.cart_pendulum.cas.dynamics import get_cart_pend_dynamics, CartPendParameters
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
from simple_planner_demo.servo_connections_planner import (
  ReducedDynamics,
  get_reduced_dynamics,
  ServoConnection,
)
from simple_planner_demo.singular_reduced_dynamics import (
  get_reduced_dynamics_positive_speed_solution,
  get_siungular_trajectory,
  plot_reduced_coefficients
)


def show_singular_phase_portrait(reduced : ReducedDynamics, singularity : tuple, 
                                  theta_left = None, theta_right = None):
  eps = 0.001
  dtheta_max = 2.0
  theta_min = reduced.diap[0]
  theta_max = reduced.diap[1]

  dtheta_singular = float(np.sqrt(-reduced.gamma(singularity) / reduced.beta(singularity)))
  plt.axvline(singularity, ls='--', color='grey')
  plt.axhline(0, ls='--', color='grey')

  for theta0 in np.linspace(singularity + eps, theta_max, 10):
    interval = [theta0, singularity + eps]
    sol = get_reduced_dynamics_positive_speed_solution(reduced, dtheta_max, interval, max_step=1e-3)
    assert sol is not None
    theta,dtheta = sol
    plt.plot(theta, dtheta, ls='-', color='blue', alpha=0.3, lw=1)
    plt.plot(theta, -dtheta, ls='-', color='blue', alpha=0.3, lw=1)
    plt.pause(0.01)

    sol = get_reduced_dynamics_positive_speed_solution(reduced, 0., interval, max_step=1e-3)
    assert sol is not None
    theta,dtheta = sol
    plt.plot(theta, dtheta, ls='-', color='blue', alpha=0.3, lw=1)
    plt.plot(theta, -dtheta, ls='-', color='blue', alpha=0.3, lw=1)
    plt.pause(0.01)

  for theta0 in np.linspace(theta_min, singularity - eps, 10):
    interval = [theta0, singularity - eps]
    sol = get_reduced_dynamics_positive_speed_solution(reduced, dtheta_max, interval, max_step=1e-3)
    assert sol is not None
    theta,dtheta = sol
    plt.plot(theta, dtheta, ls='-', color='blue', alpha=0.3, lw=1)
    plt.plot(theta, -dtheta, ls='-', color='blue', alpha=0.3, lw=1)
    plt.pause(0.01)

    sol = get_reduced_dynamics_positive_speed_solution(reduced, 0., interval, max_step=1e-3)
    assert sol is not None
    theta,dtheta = sol
    plt.plot(theta, dtheta, ls='-', color='blue', alpha=0.3, lw=1)
    plt.plot(theta, -dtheta, ls='-', color='blue', alpha=0.3, lw=1)
    plt.pause(0.01)

  for dtheta0 in np.linspace(0.3, dtheta_max, 10):
    interval = [theta_min, singularity - eps]
    sol = get_reduced_dynamics_positive_speed_solution(reduced, dtheta0, interval, max_step=1e-3)
    assert sol is not None
    theta,dtheta = sol
    plt.plot(theta, dtheta, ls='-', color='blue', alpha=0.3, lw=1)
    plt.plot(theta, -dtheta, ls='-', color='blue', alpha=0.3, lw=1)
    plt.pause(0.01)

    interval = [theta_max, singularity + eps]
    sol = get_reduced_dynamics_positive_speed_solution(reduced, dtheta0, interval, max_step=1e-3)
    assert sol is not None
    theta,dtheta = sol
    plt.plot(theta, dtheta, ls='-', color='blue', alpha=0.3, lw=1)
    plt.plot(theta, -dtheta, ls='-', color='blue', alpha=0.3, lw=1)
    plt.pause(0.01)

  # particular solution
  if theta_left is not None:
    assert theta_right
    rtraj = get_siungular_trajectory(reduced, theta_left, theta_right, singularity)
    plt.plot(rtraj.q, rtraj.dq, ls='-', color='green', alpha=1, lw=2)

  plt.plot([singularity, singularity], [-dtheta_singular, dtheta_singular], 'o', color='red')
  ax = plt.gca()
  ax.set_xlim(theta_min, theta_max)
  ax.set_ylim(-dtheta_max, dtheta_max)

  plt.xlabel(R'$\theta$')
  plt.ylabel(R'$\dot\theta$')
  plt.tight_layout()
  return plt.gcf()

def cart_pend_singular_phase_portrait():
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

  print('q1', Q_fun(0.3))
  print('q2', Q_fun(-0.3))
  print('gamma/beta', -reduced.gamma(0.) / reduced.beta(0.))
  print('beta/alpha', reduced.beta(0.) / reduced.dalpha(0.))
  print('gamma', reduced.gamma(0.))

  # plot_reduced_coefficients(reduced)
  plt.figure('cart-pendulum singular phase portrait')
  fig = show_singular_phase_portrait(reduced, 0., -0.3, 0.3)
  fig.savefig('fig/singular_phase_portrait.png', transparent=True)

def pendubot_singular_phase_portrait():
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
  q_singular = ca.vertcat(2.0, 1.2)
  v1 = 1.0 * ca.substitute(ca.pinv(mechsys.M) @ mechsys.B, mechsys.q, q_singular)
  v2 = ca.DM([0., 2.0])
  Q_expr = q_singular + v1 * theta + v2 * theta**2

  Q_fun = ca.Function('Q', [theta], [Q_expr])
  connection = ServoConnection(fun=Q_fun, diap=[-0.1, 0.1])
  reduced = get_reduced_dynamics(connection, mechsys)

  print('q1', Q_fun(0.3))
  print('q2', Q_fun(-0.3))
  print('gamma/beta', -reduced.gamma(0.) / reduced.beta(0.))
  print('beta/alpha', reduced.beta(0.) / reduced.dalpha(0.))
  print('gamma', reduced.gamma(0.))

  # plot_reduced_coefficients(reduced)
  plt.figure('pendubot singular phase portrait')
  fig = show_singular_phase_portrait(reduced, 0.)
  plt.tight_layout()
  fig.savefig('fig/singular_phase_portrait.png', transparent=True)

def type_1_singularity():
  fig = plt.figure('singularity of type I')
  plt.title(R'$\theta \ddot\theta - \dot\theta^2 + 1 = 0$')
  plt.grid(True)
  plt.axvline(0, ls='--', color='grey')

  theta0 = 0.5

  for theta0 in np.arange(0.01, 1., 0.05):
    t = np.linspace(0, 2 * np.pi * theta0, 100)
    theta = -theta0 * np.cos(t / theta0)
    dtheta = np.sin(t / theta0)
    plt.plot(theta, dtheta, alpha=0.5, lw=1, color='blue')
    plt.pause(0.01)

    t = np.linspace(0, 1, 100)
    theta = theta0 * np.sinh(t / theta0)
    dtheta = np.cosh(t / theta0)
    plt.plot(theta, dtheta, alpha=0.5, lw=1, color='blue')
    plt.plot(theta, -dtheta, alpha=0.5, lw=1, color='blue')
    plt.pause(0.01)

    t = np.linspace(0, 1, 100)
    theta = -theta0 * np.sinh(t / theta0)
    dtheta = -np.cosh(t / theta0)
    plt.plot(theta, dtheta, alpha=0.5, lw=1, color='blue')
    plt.plot(theta, -dtheta, alpha=0.5, lw=1, color='blue')
    plt.pause(0.01)

  ax = plt.gca()
  ax.set_xlim(-0.5, 0.5)
  ax.set_ylim(-2, 2)
  plt.axhline(1, color='blue', ls='-', lw=1)
  plt.axhline(-1, color='blue', ls='-', lw=1)
  
  fig.savefig('fig/type_1_singularity.png', transparent=True)

def type_2_singularity():
  fig = plt.figure('singularity of type II')
  plt.title(R'$\theta \ddot\theta + \dot\theta^2 - 1 = 0$')
  plt.grid(True)
  plt.axvline(0, ls='--', color='grey')

  theta0 = 0.5

  for theta0 in np.arange(0.01, 0.5, 0.05):
    t = np.linspace(0, 1, 100)
    dtheta0 = 0
    theta = np.sqrt(t**2 + 2*theta0*dtheta0*t + theta0**2)
    dtheta = (t + theta0*dtheta0) / np.sqrt(t**2 + 2*theta0*dtheta0*t + theta0**2)
    plt.plot(theta, dtheta, alpha=0.5, lw=1, color='blue')
    plt.plot(theta, -dtheta, alpha=0.5, lw=1, color='blue')

    t = np.linspace(0, 1, 100)
    dtheta0 = 2
    theta = np.sqrt(t**2 + 2*theta0*dtheta0*t + theta0**2)
    dtheta = (t + theta0*dtheta0) / np.sqrt(t**2 + 2*theta0*dtheta0*t + theta0**2)
    plt.plot(theta, dtheta, alpha=0.5, lw=1, color='blue')
    plt.plot(theta, -dtheta, alpha=0.5, lw=1, color='blue')

    plt.pause(0.01)
  
  for theta0 in np.arange(-0.01, -0.5, -0.05):
    t = np.linspace(0, 1, 100)
    dtheta0 = 0
    theta = -np.sqrt(t**2 + 2*theta0*dtheta0*t + theta0**2)
    dtheta = -(t + theta0*dtheta0) / np.sqrt(t**2 + 2*theta0*dtheta0*t + theta0**2)
    plt.plot(theta, dtheta, alpha=0.5, lw=1, color='blue')
    plt.plot(theta, -dtheta, alpha=0.5, lw=1, color='blue')

    t = np.linspace(0, 1, 100)
    dtheta0 = -2
    theta = -np.sqrt(t**2 + 2*theta0*dtheta0*t + theta0**2)
    dtheta = -(t + theta0*dtheta0) / np.sqrt(t**2 + 2*theta0*dtheta0*t + theta0**2)
    plt.plot(theta, dtheta, alpha=0.5, lw=1, color='blue')
    plt.plot(theta, -dtheta, alpha=0.5, lw=1, color='blue')

    plt.pause(0.01)

  ax = plt.gca()
  ax.set_xlim(-0.5, 0.5)
  ax.set_ylim(-2, 2)
  
  fig.savefig('fig/type_2_singularity.png', transparent=True)

def draw_servo_connection():
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
  
  s = np.linspace(-0.5, 0.5, 100)
  q = np.array([Q_fun(si) for si in s])[...,0]
  plt.figure(figsize=(3, 2))
  ax = plt.gca()
  plt.plot(q[:,0], q[:,1])
  an1 = ax.annotate(
    R"$Q(\theta)$",
    xy=(-1.0, 2.2),
    xycoords='data',
    xytext=(-1.0, 2.2),
    textcoords='data',
    size=18,
  )
  plt.ylabel(R'$q_2$')
  plt.xlabel(R'$q_1$')
  ax.set_xticks([], [])
  ax.set_yticks([], [])
  plt.tight_layout()
  plt.savefig('fig/servo_connection_general.png', transparent=True)

if __name__ == '__main__':
  plt.rcParams.update({
      "text.usetex": True,
      "font.family": "Helvetica",
      'font.size': 16
  })
  type_1_singularity()
  type_2_singularity()
  cart_pend_singular_phase_portrait()
  draw_servo_connection()
  pendubot_singular_phase_portrait()
