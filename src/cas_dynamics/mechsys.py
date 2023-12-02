from dataclasses import dataclass
import casadi as ca
import numpy as np
from typing import Callable


@dataclass(frozen=True, slots=True)
class MechanicalSystem:
  """
    The class contains expressions for matrices of Euler-Lagrange system
  """
  q : ca.SX | ca.MX
  dq : ca.SX | ca.MX
  u : ca.SX | ca.MX
  M : ca.SX | ca.MX
  U : ca.SX | ca.MX
  C : ca.SX | ca.MX
  G : ca.SX | ca.MX
  B : ca.SX | ca.MX

  @property
  def qdim(self) -> int:
    return self.q.shape[0]

  @property
  def udim(self) -> int:
    return self.u.shape[0]

def get_mechsys_normal_form(mechsys : MechanicalSystem) -> Callable[[float,np.ndarray,np.ndarray], np.ndarray]:
  """
    Convert symbolic expressions representing mechanical system to computable function
  """
  x = ca.vertcat(mechsys.q, mechsys.dq)
  u = mechsys.u
  M = mechsys.M
  C = mechsys.C
  G = mechsys.G
  B = mechsys.B
  Minv = ca.pinv(M)
  ddq = Minv @ (-C @ mechsys.dq - G + B @ u)
  dx = ca.vertcat(mechsys.dq, ddq)
  dynamics = ca.Function('Dynamics', [x, u], [dx])

  def rhs(_, x, u):
    dx = dynamics(x, u)
    return np.reshape(dx, (-1,))

  return rhs

def get_full_energy(mechsys : MechanicalSystem) -> ca.SX:
  dq = mechsys.dq
  U = mechsys.U
  M = mechsys.M
  E = dq.T @ M @ dq / 2 + U
  x = ca.vertcat(mechsys.q, dq)
  return ca.Function('Energy', [x], [E])

def get_lagrangian(mechsys : MechanicalSystem) -> ca.SX:
  dq = mechsys.dq
  U = mechsys.U
  M = mechsys.M
  x = ca.vertcat(mechsys.q, dq)
  L = dq.T @ M @ dq / 2 - U
  return ca.Function('Lagrangian', [x], [L])
