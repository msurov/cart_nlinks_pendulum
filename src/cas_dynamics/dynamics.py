import casadi as ca
from cas_dynamics.mechsys import MechanicalSystem
from parameters.parameters import CartPendParameters


def link_positions(q : ca.SX, par : CartPendParameters) -> list[ca.SX]:
  """
    Compute positions of mass-points and positions of joints of 
    the cart n-links pendulum system wrt world frame
  """
  x = q[0]
  joint_positions = [
    ca.horzcat(x, 0)
  ]
  mass_positions = [
    joint_positions[0]
  ]
  nlinks = par.nlinks

  for i in range(nlinks):
    theta = q[i + 1]
    vec = ca.horzcat(ca.sin(theta), ca.cos(theta))
    joint_position = joint_positions[-1] + par.link_lengths[i] * vec
    mass_position = joint_positions[-1] + par.mass_center[i] * vec
    joint_positions.append(joint_position)
    mass_positions.append(mass_position)

  return mass_positions, joint_positions

def kinetic_energy_mat(q : ca.SX, par : CartPendParameters) -> ca.SX:
  """
    Compute expression for kinetic energy matrix
  """
  positions,_ = link_positions(q, par)
  dim,_ = q.shape
  M = ca.SX.zeros(dim, dim)
  for p,m in zip(positions, par.masses):
    Jp = ca.jacobian(p, q)
    M += m * Jp.T @ Jp
  return M

def potential_energy(q : ca.SX, par : CartPendParameters) -> ca.SX:
  """
    Compute expression for potential energy
  """
  positions,_ = link_positions(q, par)
  U = 0
  for p,m in zip(positions, par.masses):
    U += m * p[1] * par.gravity_accel
  return U

def eval_C(M : ca.SX, q : ca.SX, dq : ca.SX):
  Mdq = M @ dq
  DMdq = ca.jacobian(Mdq, q)
  C = DMdq - DMdq.T / 2
  return C

def get_cart_pend_dynamics(par : CartPendParameters, simplify : bool = False) -> MechanicalSystem:
  """
    Compute expressions for the matrices of the cart-pendulum mechanical system
  """
  nlinks = par.nlinks
  u = ca.SX.sym('u')
  q = ca.SX.sym('q', nlinks + 1)
  dq = ca.SX.sym('dq', nlinks + 1)

  M = kinetic_energy_mat(q, par)
  U = potential_energy(q, par)
  C = eval_C(M, q, dq)
  G = ca.jacobian(U, q).T
  B = ca.SX.zeros(nlinks + 1, 1)
  B[0] = 1

  return MechanicalSystem(
    q = q,
    dq = dq,
    u = u,
    U = U,
    M = M,
    C = C,
    G = G,
    B = B,
  )
