import casadi as ca
from dynamics.cas_mechsys import MechanicalSystem
from ..parameters import DoublePendulumParameters


def get_mass_positions(q : ca.SX, par : DoublePendulumParameters):
  p1 = par.mass_center_1 * ca.vertcat(ca.sin(q[0]), ca.cos(q[0]))
  p2 = par.link_length_1 * ca.vertcat(ca.sin(q[0]), ca.cos(q[0])) + \
        par.mass_center_2 * ca.vertcat(ca.sin(q[1]), ca.cos(q[1]))
  return [p1, p2]

def get_kinetic_energy_mat(q : ca.SX, par : DoublePendulumParameters):
  p1, p2 = get_mass_positions(q, par)
  Jp1 = ca.jacobian(p1, q)
  Jp2 = ca.jacobian(p2, q)
  M = Jp1.T @ Jp1 * par.mass_1 + Jp2.T @ Jp2 * par.mass_2
  return M

def eval_C(M : ca.SX, q : ca.SX, dq : ca.SX):
  Mdq = M @ dq
  DMdq = ca.jacobian(Mdq, q)
  C = DMdq - DMdq.T / 2
  return C

def get_potential_energy(q : ca.SX, par : DoublePendulumParameters):
  p1, p2 = get_mass_positions(q, par)
  U = p1[1] * par.mass_1 * par.gravity_accel + \
      p2[1] * par.mass_2 * par.gravity_accel
  return U

def get_double_pendulum_dynamics(par : DoublePendulumParameters):
  q = ca.SX.sym('q', 2)
  dq = ca.SX.sym('q', 2)
  u = ca.SX.sym('u')

  M = get_kinetic_energy_mat(q, par)
  U = get_potential_energy(q, par)
  G = ca.jacobian(U, q).T
  C = eval_C(M, q, dq)
  
  B = ca.DM.zeros(2)
  B[par.actuated_joint] = 1

  return MechanicalSystem(
    q = q,
    dq = dq,
    u = u,
    U = U,
    M = M,
    C = C,
    G = G,
    B = B
  )
