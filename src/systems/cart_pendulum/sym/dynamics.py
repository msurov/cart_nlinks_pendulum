import sympy as sy
from ....dynamics.sym_mechsys import MechanicalSystem
from parameters.parameters import CartPendParameters


def link_positions(q : sy.Tuple, par : CartPendParameters) -> list[sy.Matrix]:
  """
    Compute positions of mass-points and positions of joints of 
    the cart n-links pendulum system wrt world frame
  """
  x = q[0]
  joint_positions = [sy.Matrix([[x], [0]])]
  mass_positions = [joint_positions[0]]
  nlinks = par.nlinks

  for i in range(nlinks):

    theta = q[i + 1]
    vec = sy.Matrix([
        [sy.sin(theta)],
        [sy.cos(theta)]
      ])
    joint_position = joint_positions[-1] + par.link_lengths[i] * vec
    mass_position = joint_positions[-1] + par.mass_center[i] * vec
    joint_positions.append(joint_position)
    mass_positions.append(mass_position)

  return mass_positions, joint_positions

def kinetic_energy_mat(q : sy.Tuple, par : CartPendParameters) -> sy.Matrix:
  """
    Compute expression for kinetic energy matrix
  """
  positions,_ = link_positions(q, par)
  dim = len(q)
  M = sy.zeros(dim, dim)
  for p,m in zip(positions, par.masses):
    Jp = p.jacobian(q)
    M += m * Jp.T @ Jp
  return M

def potential_energy(q : sy.Tuple, par : CartPendParameters) -> sy.Expr:
  """
    Compute expression for potential energy
  """
  positions,_ = link_positions(q, par)
  U = 0
  for p,m in zip(positions, par.masses):
    U += m * p[1] * par.gravity_accel
  return U

def eval_christ(q : sy.Tuple, M : sy.Matrix) -> sy.Array:
  """
    Compute Christoffel symbols
  """
  DM = sy.derive_by_array(M, q)
  D1 = sy.permutedims(DM, index_order_old="ijk", index_order_new= "kij")
  D2 = sy.permutedims(DM, index_order_old="ijk", index_order_new= "jik")
  D3 = DM
  Christ = (D1 + D2 - D3) / 2
  return Christ

def eval_C(q : sy.Tuple, dq : sy.Matrix, M : sy.Matrix) -> sy.Matrix:
  """
    Compute expression for Coriolis forces matrix
  """
  Christ = eval_christ(q, M)
  C = sy.tensorcontraction(
    sy.tensorproduct(Christ, dq),
    (1, 3)
  )
  return C.tomatrix()

def get_cart_pend_dynamics(par : CartPendParameters, simplify : bool = False) -> MechanicalSystem:
  """
    Compute expressions for the matrices of the cart-pendulum mechanical system
  """
  nlinks = par.nlinks
  u = sy.Symbol('u', real=True)
  q = sy.symbols(Rf'x \theta_(1:{nlinks + 1})', real=True)
  q = sy.Tuple(*q)
  dq = sy.symbols(R'\dot{x} \dot{\theta}_(1:' + str(nlinks + 1) + ')', real=True)

  M = kinetic_energy_mat(q, par)
  if simplify:
    M = sy.simplify(M)
  U = potential_energy(q, par)
  if simplify:
    U = sy.simplify(U)
  C = eval_C(q, dq, M)
  if simplify:
    C = sy.simplify(C)
  G = sy.ImmutableMatrix([U]).jacobian(q).T
  if simplify:
    G = sy.simplify(G)
  B = sy.zeros(nlinks + 1, 1)
  B[0] = 1

  return MechanicalSystem(
    q = q,
    dq = sy.ImmutableMatrix(dq),
    u = sy.ImmutableMatrix([u]),
    U = U,
    M = sy.ImmutableMatrix(M),
    C = sy.ImmutableMatrix(C),
    G = sy.ImmutableMatrix(G),
    B = sy.ImmutableMatrix(B),
  )
