from sym_dynamics.dynamics import MechanicalSystem
import sympy as sy


def get_subsystem(dynamics : MechanicalSystem, simplify=False) -> MechanicalSystem:
  M = dynamics.M
  C = dynamics.C
  G = dynamics.G
  B = dynamics.B
  u = dynamics.u
  q = dynamics.q
  dq = dynamics.dq

  m11 = M[0,0]
  m12 = M[0,1:]
  m21 = M[1:,0]
  m22 = M[1:,1:]

  # @todo: be sure this is correct
  C = C.subs(dq[0], 0)
  c1 = C[0,1:]
  c2 = C[1:,1:]

  g1 = G[0,:]
  g2 = G[1:,:]

  b1 = B[0,:]
  b2 = B[1:,:]

  Mr = m22 * m11 - m21 @ m12
  Cr = m11 * c2 - m21 * c1
  Gr = m11 * g2 - m21 * g1
  Br = m11 * b2 - m21 * b1

  if simplify:
    Mr = sy.simplify(Mr)
    Cr = sy.simplify(Cr)
    Gr = sy.simplify(Gr)
    Br = sy.simplify(Br)

  return MechanicalSystem(
      u = dynamics.u,
      q = q[1:],
      dq = dq[1:,0],
      M = Mr,
      C = Cr,
      G = Gr,
      B = Br,
      U = dynamics.U
  )
