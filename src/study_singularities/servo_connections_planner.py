from motion_planner.trajectory import MechanicalSystemTrajectory
from dynamics.cas_mechsys import MechanicalSystem, solve_for_control_inputs
import casadi as ca
from dataclasses import dataclass
from common.integrate import solve_periodic_ivp
import numpy as np
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

def get_reduced_dynamics(servo_connection : ServoConnection, mechsys : MechanicalSystem) -> ReducedDynamics:
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

def get_reduced_periodic_solution(reduced : ReducedDynamics, initial : np.ndarray, period_max) -> MechanicalSystemTrajectory:
  def rhs(t, st):
    s, ds = st
    dds = (-reduced.beta(s)*ds**2 - reduced.gamma(s)) / reduced.alpha(s)
    dds = float(dds)
    return np.array([ds, dds])
  sol = solve_periodic_ivp(rhs, 0., initial, 1e-2, period_max, max_step=1e-3)
  ddq = np.array([rhs(ti, xi)[1] for ti,xi in zip(sol.t, sol.x)])

  return MechanicalSystemTrajectory(
    t = sol.t,
    q = sol.x[:,0],
    dq = sol.x[:,1],
    ddq = ddq,
    u = None
  )

def get_original_system_trajectory(reduced_traj : MechanicalSystemTrajectory, connection : ServoConnection, mechsys : MechanicalSystem) -> MechanicalSystemTrajectory:
  nq = mechsys.qdim
  nt, = reduced_traj.t.shape
  _,nu = mechsys.B.shape

  q = np.zeros((nt, nq))
  dq = np.zeros((nt, nq))
  ddq = np.zeros((nt, nq))
  u = np.zeros((nt, nu))
  u_fun = solve_for_control_inputs(mechsys)

  for i in range(nt):
    theta = reduced_traj.q[i]
    dtheta = reduced_traj.dq[i]
    ddtheta = reduced_traj.ddq[i]
    q[i,:] = np.reshape(connection.fun(theta), (-1,))
    dq[i,:] = np.reshape(connection.first_deriv(theta) * dtheta, (-1,))
    ddq[i,:] = np.reshape(connection.first_deriv(theta) * ddtheta + connection.second_deriv(theta) * dtheta**2, (-1,))
    u[i,:] = np.reshape(u_fun(q[i,:], dq[i,:], ddq[i,:]), (-1,))

  return MechanicalSystemTrajectory(t = reduced_traj.t, q = q, dq = dq, ddq = ddq, u = u)
