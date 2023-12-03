from cas_dynamics.dynamics import (
  CartPendParameters,
  get_cart_pend_dynamics,
  MechanicalSystem
)
from motion_planner.trajectory import MechanicalSystemTrajectory
from motion_planner.study_singularities.planner_v2 import find_periodic_traj
from dataclasses import dataclass
import casadi as ca
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.integrate import solve_ivp
from casadi_bspline.bspline import get_spline_symexpr
from scipy.optimize import fsolve, brentq
import matplotlib.pyplot as plt


@dataclass(frozen=True, slots=True)
class ReducedDynamics:
  alpha : ca.Function
  dalpha : ca.Function
  beta : ca.Function
  gamma : ca.Function
  dgamma : ca.Function
  diap : tuple[float]

@dataclass(slots=True)
class ServoConnection:
  fun : ca.Function
  diap : tuple[float]

  first_deriv : ca.Function = None
  second_deriv : ca.Function = None

  def __post_init__(self):
    s = ca.SX.sym('arg')

    if self.first_deriv is None:
      tmp = ca.jacobian(self.fun(s), s)
      self.first_deriv = ca.Function('servo_connection_first_deriv', [s], [tmp])
    
    if self.second_deriv is None:
      tmp = ca.jacobian(self.first_deriv(s), s)
      self.second_deriv = ca.Function('servo_connection_first_deriv', [s], [tmp])

def get_reduced_dynamics(servo_connection : ServoConnection, mechsys : MechanicalSystem):
  B = mechsys.B
  B_perp = B.T @ ca.DM([[0, -1], [1, 0]])
  C = mechsys.C
  M = mechsys.M
  G = mechsys.G
  q = mechsys.q
  dq = mechsys.dq
  s = ca.SX.sym('s')
  Q = servo_connection.fun(s)
  dQ = servo_connection.first_deriv(s)
  ddQ = servo_connection.second_deriv(s)

  B_perp_M = ca.substitute(B_perp @ M, q, Q)

  alpha = B_perp_M @ dQ
  beta = B_perp_M @ ddQ + ca.substitute(B_perp @ C, ca.vertcat(q, dq), ca.vertcat(Q, dQ)) @ dQ
  gamma = ca.substitute(B_perp @ G, q, Q)
  dalpha = ca.jacobian(alpha, s)
  dgamma = ca.jacobian(gamma, s)
  reduced = ReducedDynamics(
    alpha = ca.Function('alpha', [s], [alpha]),
    dalpha = ca.Function('dalpha', [s], [dalpha]),
    beta = ca.Function('beta', [s], [beta]),
    gamma = ca.Function('gamma', [s], [gamma]),
    dgamma = ca.Function('dgamma', [s], [dgamma]),
    diap = servo_connection.diap
  )
  return reduced

def curve_legth(q, weights=None):
  dq = np.diff(q, axis=0)
  _,dim = np.shape(q)
  if weights is None:
    weights = np.ones((dim, 1))
  ds = np.sqrt(np.sum(dq**2 * weights, axis=1))
  s = np.cumsum(ds)
  s = np.concatenate([[0], s])
  return s

def get_trajectory_servo_connection(traj : MechanicalSystemTrajectory, mechsys : MechanicalSystem):
  weights = np.array([1., 1.])
  s = curve_legth(traj.q, weights)

  sp = make_interp_spline(s, traj.q, k=5, bc_type='periodic')
  arg = ca.SX.sym('servo_connection_arg')
  expr = get_spline_symexpr(sp, arg)
  expr1 = ca.jacobian(expr, arg)
  expr2 = ca.jacobian(expr1, arg)
  connection = ServoConnection(
    fun = ca.Function('servo_connection_fun', [arg], [expr.T]),
    first_deriv = ca.Function('servo_connection_first_deriv', [arg], [expr1]),
    second_deriv = ca.Function('servo_connection_second_deriv', [arg], [expr2]),
    diap = (s[0], s[-1])
  )
  return connection

def get_reduced_dynamics_singularities(reduced : ReducedDynamics):
  def fun(x):
    alpha = reduced.alpha(x)
    alpha = np.reshape(np.array(alpha, float), np.shape(x))
    return alpha

  knots = np.linspace(*reduced.diap, 100)
  alpha = reduced.alpha(knots)
  alpha = np.array(alpha, float)[:,0]
  b = alpha[:-1] * alpha[1:] <= 0
  indices, = np.nonzero(b)
  roots = []

  for i in indices:
    x1 = knots[i]
    x2 = knots[i + 1]
    res = brentq(fun, x1, x2, maxiter=1000, full_output=True)
    assert res[1].converged, f"Can't refine root between {x1} and {x2}"
    roots.append(res[0])

  return roots

def plot_reduced_coefficients(reduced : ReducedDynamics):
  singularities = get_reduced_dynamics_singularities(reduced)
  s1,s2 = reduced.diap
  s = np.linspace(s1, s2, 600)
  alpha = np.zeros(len(s), float)
  dalpha = np.zeros(len(s), float)
  beta = np.zeros(len(s), float)
  gamma = np.zeros(len(s), float)
  dgamma = np.zeros(len(s), float)
  for i in range(len(s)):
    alpha[i] = float(reduced.alpha(s[i]))
    dalpha[i] = float(reduced.dalpha(s[i]))
    beta[i] = float(reduced.beta(s[i]))
    gamma[i] = float(reduced.gamma(s[i]))
    dgamma[i] = float(reduced.dgamma(s[i]))

  ax = plt.subplot(321)
  plt.grid(True)
  plt.axhline(0, ls='--', color='grey')
  [plt.axvline(sing, ls='--', color='red') for sing in singularities]
  plt.ylabel(R'$\alpha$')
  plt.plot(s, alpha)

  plt.subplot(323, sharex=ax)
  plt.grid(True)
  [plt.axvline(sing, ls='--', color='red') for sing in singularities]
  plt.axhline(0, ls='--', color='grey')
  plt.ylabel(R'$\beta$')
  plt.plot(s, beta)

  plt.subplot(325, sharex=ax)
  plt.grid(True)
  [plt.axvline(sing, ls='--', color='red') for sing in singularities]
  plt.axhline(0, ls='--', color='grey')
  plt.ylabel(R'$\gamma$')
  plt.plot(s, gamma)

  plt.subplot(322, sharex=ax)
  plt.grid(True)
  [plt.axvline(sing, ls='--', color='red') for sing in singularities]
  plt.axhline(0, ls='--', color='grey')
  plt.ylabel(R'$\dot\alpha$')
  plt.plot(s, dalpha)

  plt.subplot(324, sharex=ax)
  plt.grid(True)
  [plt.axvline(sing, ls='--', color='red') for sing in singularities]
  plt.axhline(0, ls='--', color='grey')
  plt.ylabel(R'$\frac{\beta}{\dot \alpha}$')
  plt.plot(s, beta/dalpha)

  plt.subplot(326, sharex=ax)
  plt.grid(True)
  [plt.axvline(sing, ls='--', color='red') for sing in singularities]
  plt.axhline(0, ls='--', color='grey')
  plt.ylabel(R'$\frac{\gamma}{\beta}$')
  plt.plot(s, gamma/beta)

  plt.show()

def show_phase_portrait(reduced : ReducedDynamics, theta_singular : float, theta_diap : tuple[float], dtheta_max : float):
  def rhs(x, y):
    dy = (-2*reduced.beta(x)*y - reduced.gamma(x)) / reduced.alpha(x)
    return float(dy)
  
  ymax = dtheta_max**2 / 2
  plt.axvline(theta_singular, ls='--', color='grey')
  dtheta_singular = np.sqrt(-reduced.gamma(theta_singular) / reduced.beta(theta_singular))
  plt.plot(theta_singular, dtheta_singular, 'o', alpha=0.5, color='red')
  plt.pause(0.01)

  for y0 in np.linspace(0, ymax, 10):
    sol = solve_ivp(rhs, theta_diap, [y0], max_step=1e-3)
    dtheta = np.sqrt(2 * sol.y[0])
    theta = sol.t
    plt.plot(theta, dtheta, ls='--', alpha=0.5, color='lightblue')
    plt.plot(theta, -dtheta, ls='--', alpha=0.5, color='lightblue')
    plt.pause(0.01)

def demo():
  par = CartPendParameters(
    link_lengths=[2],
    mass_center=[1],
    masses=[1, 1],
    gravity_accel=1
  )
  mechsys = get_cart_pend_dynamics(par)
  t = ca.SX.sym('t')
  theta_ref = ca.Function('theta_ref', [t], [0.6 * ca.sin(t) - 0.4 * ca.sin(2*t)])
  traj = find_periodic_traj(theta_ref, 2 * np.pi, mechsys)
  connection = get_trajectory_servo_connection(traj, mechsys)
  # s = np.linspace(*connection.diap, 1000)
  # q = np.array([connection.fun(si) for si in s])[:,:,0]
  # plt.plot(s, q)
  # plt.show()
  reduced = get_reduced_dynamics(connection, mechsys)
  # plot_reduced_coefficients(reduced)
  singularities = get_reduced_dynamics_singularities(reduced)

  theta0 = connection.diap[0]
  dq = traj.dq[0]
  dQ = connection.first_deriv(theta0)
  dtheta0 = (dQ.T @ dq) / (dQ.T @ dQ)
  dtheta0 = float(dtheta0)
  print('dtheta0:', dtheta0)
  theta_sing = singularities[0]
  thetadiap = [theta0, theta_sing - 0.01]
  show_phase_portrait(reduced, singularities[0], thetadiap, dtheta0)
  plt.show()

if __name__ == '__main__':
  demo()
