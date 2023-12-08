import matplotlib.pyplot as plt
import numpy as np
from motion_planner.trajectory import MechanicalSystemTrajectory
from simple_planner_demo.servo_connections_planner import (
  ReducedDynamics,
)
from scipy.integrate import solve_ivp


def get_trajectory_time(theta : np.ndarray, dtheta : np.ndarray):
  delta_theta = np.diff(theta)
  mdtheta = (dtheta[1:] + dtheta[:-1]) / 2
  delta_t = delta_theta / mdtheta
  t = np.cumsum(delta_t)
  return np.concatenate(([0], t))

def get_reduced_dynamics_positive_speed_solution(
    reduced : ReducedDynamics,
    dtheta0 : float,
    theta_interval : tuple[float], **integrator_args):

  def rhs(x, y):
    dy = (-2*reduced.beta(x) * y - reduced.gamma(x)) / reduced.alpha(x)
    return float(dy)

  y0 = dtheta0**2 / 2
  sol = solve_ivp(rhs, theta_interval, [y0], **integrator_args)
  if not sol.success:
    return None
  dtheta = np.sqrt(2*sol.y[0])
  return sol.t, dtheta

def get_siungular_trajectory(reduced : ReducedDynamics, theta_left : float, theta_right : float, singularity : float):
  eps = 1e-3
  dtheta_singular = float(np.sqrt(-reduced.gamma(singularity) / reduced.beta(singularity)))
  theta1,dtheta1 = get_reduced_dynamics_positive_speed_solution(reduced, 0., [theta_left, singularity - eps], max_step=1e-3)
  theta2,dtheta2 = get_reduced_dynamics_positive_speed_solution(reduced, 0., [theta_right, singularity + eps], max_step=1e-3)
  theta12 = np.concatenate((theta1, [singularity], theta2[::-1]))
  dtheta12 = np.concatenate((dtheta1, [dtheta_singular], dtheta2[::-1]))
  theta = np.concatenate((theta12, theta12[-2::-1]))
  dtheta = np.concatenate((dtheta12, -dtheta12[-2::-1]))
  ddtheta = (-reduced.beta(theta) * dtheta**2 - reduced.gamma(theta)) / reduced.alpha(theta)
  ddtheta = np.array(ddtheta)

  t = get_trajectory_time(theta, dtheta)
  return MechanicalSystemTrajectory(
    t = t,
    q = theta,
    dq = dtheta,
    ddq = ddtheta,
    u = None
  )

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
