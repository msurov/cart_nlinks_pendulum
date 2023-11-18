from typing import Callable
import numpy as np

def lorenz_attractor(sigma : float, rho : float, beta : float) -> Callable[[float, np.ndarray, np.ndarray], np.ndarray]:
  def sys(t, st, u):
    x,y,z = st
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y  - beta * z + u[0]
    return np.array([dx, dy, dz])
  return sys
