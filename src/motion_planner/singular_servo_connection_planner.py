#%%
from sym_dynamics.mechsys import MechanicalSystem
from motion_planner.subsystem import get_subsystem
from sym_dynamics.dynamics import (
  get_cart_pend_dynamics,
  CartPendParameters
)
from common.integrate import (
  solve_periodic_ivp,
  integrate_twice
)
from motion_planner.trajectory import (
  MechanicalSystemTrajectory,
  MechanicalSystemPeriodicTrajectory
)
from IPython.display import display
from dataclasses import dataclass
import numpy as np
import sympy as sy
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
from scipy.optimize import brentq


@dataclass(frozen=True, slots=True)
class SymFunction:
  expr : sy.Tuple
  arg : sy.Symbol

@dataclass(frozen=True, slots=True)
class ReducedDyanmics:
  arg : sy.Symbol
  alpha : sy.Expr
  beta : sy.Expr
  gamma : sy.Expr


def find_all_roots(fun, diap, step):
  a,b = diap
  npts = int((b - a + step) / step)
  pts = np.linspace(a, b, npts)
  vals = np.array([fun(e) for e in pts])
  print('vals:', vals)
  intervals, = np.nonzero(vals[:-1] * vals[1:] <= 0)
  return np.array([brentq(fun, pts[i], pts[i+1]) for i in intervals])

def apply_servo_connection(
    sys : MechanicalSystem,
    servo_connection : SymFunction
  ) -> ReducedDyanmics:
  """
  """
  Q = sy.Matrix(servo_connection.expr)
  dQ = Q.diff(servo_connection.arg)
  ddQ = dQ.diff(servo_connection.arg)

  M = sys.M.subs(zip(sys.q, Q))
  C = sys.C.subs(zip(sys.dq, dQ))
  C = C.subs(zip(sys.q, Q))
  G = sys.G.subs(zip(sys.q, Q))
  B = sys.B.subs(zip(sys.q, Q))
  B_perp = sy.Matrix([[B[1,0], -B[0,0]]])

  alpha = (B_perp @ M @ dQ)[0,0]
  beta = (B_perp @ (M @ ddQ + C @ dQ))[0,0]
  gamma = (B_perp @ G)[0,0]

  alpha = sy.simplify(alpha)
  beta = sy.simplify(beta)
  gamma = sy.simplify(gamma)

  return ReducedDyanmics(
    arg = servo_connection.arg,
    alpha = alpha,
    beta = beta,
    gamma = gamma
  )

def test():
  rat = sy.Rational
  par = CartPendParameters(
    link_lengths = [1, 1],
    mass_center = [rat(1, 2), rat(1, 2)],
    masses = [rat(1, 8), rat(1, 8), rat(1, 8)],
    gravity_accel = 10
  )
  sys = get_cart_pend_dynamics(par, simplify=True)
  subsys = get_subsystem(sys, simplify=True)
  B_perp = sy.Matrix([[subsys.B[1,0], -subsys.B[0,0]]])

  phi = sy.Symbol(R'\phi', real=True)
  connection = SymFunction(
    expr = sy.Tuple(
      rat(21, 10) - rat(3, 8) * (phi - rat(25, 10)),
      phi
    ),
    arg = phi
  )

  reduced_dynamics = apply_servo_connection(subsys, connection)
  dalpha = sy.lambdify(reduced_dynamics.arg, reduced_dynamics.alpha.diff(reduced_dynamics.arg))
  alpha = sy.lambdify(reduced_dynamics.arg, reduced_dynamics.alpha)
  beta = sy.lambdify(reduced_dynamics.arg, reduced_dynamics.beta)
  gamma = sy.lambdify(reduced_dynamics.arg, reduced_dynamics.gamma)
  dgamma = sy.lambdify(reduced_dynamics.arg, reduced_dynamics.gamma.diff(reduced_dynamics.arg))
  phi = np.linspace(0, np.pi, 100)

  alpha_val = [alpha(e) for e in phi]
  gamma_val = [gamma(e) for e in phi]
  beta_val = [beta(e) for e in phi]

  alpha_roots = find_all_roots(alpha, [0, np.pi], 0.1)
  print(alpha_roots)
  gamma_roots = find_all_roots(gamma, [0, np.pi], 0.1)

  for s in alpha_roots:
    theta1,theta2 = connection.expr.subs(connection.arg, s)
    print(f'# singular point at {s}: {theta1, theta2}')
    print(f'  -gam/bet: {-gamma(s) / beta(s)}')
    print(f'  dalpha: {dalpha(s)}')
    print(f'  gam: {gamma(s)}')

  for s in gamma_roots:
    theta1,theta2 = connection.expr.subs(connection.arg, s)
    print(f'# stationary point at {s}: {theta1, theta2}')
    print(f'  -dgamma/alpha: {-dgamma(s) / alpha(s)}')

  ax = plt.subplot(311)
  plt.grid(True)
  plt.plot(phi, alpha_val)
  plt.ylabel('alpha')

  ax = plt.subplot(312, sharex=ax)
  plt.grid(True)
  plt.plot(phi, beta_val)
  plt.ylabel('beta')

  ax = plt.subplot(313, sharex=ax)
  plt.grid(True)
  plt.plot(phi, gamma_val)
  plt.ylabel('gamma')

  plt.show()



if __name__ == '__main__':
  test()

#%%