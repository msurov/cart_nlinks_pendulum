from dataclasses import dataclass
import sympy as sy
import numpy as np
from typing import Callable


@dataclass(frozen=True, slots=True)
class MechanicalSystem:
  """
    The class contains expressions for matrices of Euler-Lagrange system
  """
  q : sy.Tuple
  dq : sy.Matrix
  u : sy.Matrix
  M : sy.Matrix
  U : sy.Expr
  C : sy.Matrix
  G : sy.Matrix
  B : sy.Matrix


def get_numeric_dynamics(mechsys : MechanicalSystem) -> Callable[[float,np.ndarray,np.ndarray],np.ndarray]:
  """
    Convert symbolic expressions representing mechanical system to computable function
  """
  x = sy.Tuple(*mechsys.q, *mechsys.dq)
  dq = mechsys.dq
  u = mechsys.u
  M = mechsys.M
  C = mechsys.C
  G = mechsys.G
  B = mechsys.B
  Minv = M.inv()
  ddq = Minv @ (-C @ dq - G + B @ u)
  dx = sy.Tuple(*dq, *ddq)
  rhs = sy.lambdify((x, u), dx, 'numpy', )

  def dynamical_system(time : float, system_state : np.ndarray, system_input : np.ndarray):
    d_system_state = rhs(system_state, system_input)
    return np.array(d_system_state)

  return dynamical_system

def get_full_energy(mechsys : MechanicalSystem) -> sy.Expr:
  dq = mechsys.dq
  U = mechsys.U
  M = mechsys.M
  return (dq.T @ M @ dq / 2)[0,0] + U

def get_lagrangian(mechsys : MechanicalSystem) -> sy.Expr:
  dq = mechsys.dq
  U = mechsys.U
  M = mechsys.M
  return (dq.T @ M @ dq / 2)[0,0] - U
