from sym_dynamics.mechsys import MechanicalSystem
from motion_planner.subsystem import get_subsystem
from sym_dynamics.cart_pend_dynamics import (
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
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from common.symfunc import SymFunction


@dataclass(frozen=True, slots=True)
class ReducedDyanmics:
  arg : sy.Symbol
  alpha : sy.Expr
  beta : sy.Expr
  gamma : sy.Expr


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


def compute_zero_dynamics_trajectory(
    zero_dynamics : ReducedDyanmics,
    initial_position : np.ndarray,
    duration : float,
    step : float = 0.01
  ) -> MechanicalSystemPeriodicTrajectory:
  """
  """
  alpha = zero_dynamics.alpha
  beta = zero_dynamics.beta
  gamma = zero_dynamics.gamma
  speed = sy.Dummy('d' + zero_dynamics.arg.name)
  rhs_expr = sy.Tuple(speed, (-beta * speed**2 - gamma) / alpha)
  rhs_fun = sy.lambdify((zero_dynamics.arg, speed), rhs_expr)

  def sys(t, st):
    dst = rhs_fun(*st)
    return np.array(dst)
  sol = solve_periodic_ivp(sys, 0., [initial_position, 0.], step, 
                     duration, max_step=1e-3, atol=1e-9, rtol=1e-9, nsteps=100)
  ddx = np.array([rhs_fun(*e)[1] for e in sol.x])
  return MechanicalSystemPeriodicTrajectory(
    t = sol.t,
    q = sol.x[:,0],
    dq = sol.x[:,1],
    ddq = ddx,
    u = None,
    period = sol.period,
    periodicity_deviation = sol.periodicity_deviation
  )

def get_inv_dynamics(sys : MechanicalSystem):
  qdim = sys.qdim
  name = R'\ddot{q}_' + f'(0:{qdim})'
  ddq_ = sy.symbols(name, real=True)
  ddq = sy.Matrix(ddq_)
  u_expr = sys.B.pinv() @ (sys.M @ ddq + sys.C @ sys.dq + sys.G)
  return sy.lambdify(
    (sys.q, sy.Tuple(*sys.dq), ddq_),
    u_expr
  )

def get_original_system_trajectory(
    zero_dynamics_traj : MechanicalSystemPeriodicTrajectory,
    servo_connection : SymFunction,
    sys : MechanicalSystem
  ) -> MechanicalSystemPeriodicTrajectory:
  R"""
    Construct periodic trajectory of original mechanical system 
    from the given reduced dynamics periodic trajectory
  """
  phi = servo_connection.arg
  Q_expr = sy.Matrix(servo_connection.expr)
  dQ_expr = Q_expr.diff(phi)
  ddQ_expr = dQ_expr.diff(phi)
  Q_fun = sy.lambdify(servo_connection.arg, Q_expr)
  dQ_fun = sy.lambdify(servo_connection.arg, dQ_expr)
  ddQ_fun = sy.lambdify(servo_connection.arg, ddQ_expr)
  u_fun = get_inv_dynamics(sys)

  qdim = sys.qdim
  udim = sys.udim
  npts = len(zero_dynamics_traj.t)
  q = np.zeros((npts, qdim))
  dq = np.zeros((npts, qdim))
  ddq = np.zeros((npts, qdim))
  u = np.zeros((npts, udim))

  for i in range(npts):
    phi = zero_dynamics_traj.q[i]
    dphi = zero_dynamics_traj.dq[i]
    ddphi = zero_dynamics_traj.ddq[i]
    q[i,:] = Q_fun(phi)[:,0]
    dq[i,:] = dQ_fun(phi)[:,0] * dphi
    ddq[i,:] = ddQ_fun(phi)[:,0] * dphi**2 + dQ_fun(phi)[:,0] * ddphi
    u[i,:] = u_fun(q[i], dq[i], ddq[i])
  
  return MechanicalSystemPeriodicTrajectory(
    t = zero_dynamics_traj.t,
    q = q,
    dq = dq,
    ddq = ddq,
    u = u,
    period = zero_dynamics_traj.period,
    periodicity_deviation = None
  )

def find_sample_trajectory(par : CartPendParameters):
  rat = sy.Rational
  sys = get_cart_pend_dynamics(par, simplify=True)
  subsys = get_subsystem(sys, simplify=True)

  phi = sy.Symbol(R'\phi', real=True)
  # k = sy.sympify(4) / 3
  # connection = SymFunction(
  #   expr = sy.Tuple(k * phi, phi),
  #   arg = phi
  # )
  connection = SymFunction(
    expr = sy.Tuple(
      rat(21, 10) - rat(3, 8) * (phi - rat(25, 10)),
      phi
    ),
    arg = phi
  )
  phi0 = 2.2

  zero_dynamics = apply_servo_connection(subsys, connection)

  # stationary point
  gamma_fun = sy.lambdify(zero_dynamics.arg, zero_dynamics.gamma)
  phi_eq = brentq(gamma_fun, phi0 - 0.1, phi0 + 0.1)

  assert abs(gamma_fun(phi_eq)) < 1e-5, 'There is not equilibrium at zero'
  dgamma = zero_dynamics.gamma.diff(zero_dynamics.arg)
  lin = -dgamma / zero_dynamics.alpha
  assert lin.subs(zero_dynamics.arg, phi_eq) < 0, 'Equilibrium is not stable'

  # compute zero dynamics trajectory
  zero_dynamics_traj = compute_zero_dynamics_trajectory(zero_dynamics, phi_eq + 0.1, 5.0, step=1e-3)
  traj = get_original_system_trajectory(zero_dynamics_traj, connection, subsys)

  ddx_expr = (sys.M.inv() @ (-sys.C @ sys.dq - sys.G + sys.B @ sys.u))[0,0]
  ddx_expr = sy.simplify(ddx_expr)
  ddx_fun = sy.lambdify(
    (subsys.q, subsys.dq, subsys.u),
    ddx_expr
  )
  npts = len(traj.t)
  ddx = np.zeros(npts)
  for i in range(npts):
    ddx[i] = ddx_fun(traj.q[i], traj.dq[i], traj.u[i])
  sp = make_interp_spline(traj.t, ddx, k=3, bc_type='periodic')
  x, dx = integrate_twice(sp, traj.t)

  plt.plot(traj.t, x)
  plt.plot(traj.t, dx)
  plt.show()
  exit()

  q = np.zeros((npts, sys.qdim))
  q[:,0] = x
  q[:,1:] = traj.q
  dq = np.zeros((npts, sys.qdim))
  dq[:,0] = dx
  dq[:,1:] = traj.dq
  ddq = np.zeros((npts, sys.qdim))
  ddq[:,0] = ddx
  ddq[:,1:] = traj.ddq

  return MechanicalSystemPeriodicTrajectory(
    t = traj.t,
    q = q,
    dq = dq,
    ddq = ddq,
    u = traj.u,
    period = traj.period,
    periodicity_deviation = None
  )
