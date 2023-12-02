from scipy.integrate import ode
import numpy as np
from copy import copy
from dataclasses import dataclass
from typing import Callable


@dataclass
class SimulationResult:
  t : np.ndarray
  x : np.ndarray
  u : np.ndarray
  state : list

def simulate(
    sys : Callable[[float, np.ndarray, np.ndarray], np.ndarray],
    fb : Callable[[float, np.ndarray], np.ndarray],
    tspan : tuple,
    xstart : np.ndarray,
    step : float,
    **integrator_args
  ) -> SimulationResult:
  R"""
  :param sys: sys(t, x, u) -> dx
  :param fb: fb(t, x) -> u
  :param tspan: (tstart, tend)
  :return: SimulationResult
  """

  t = tstart = float(tspan[0])
  tend = float(tspan[1])
  x = np.array(xstart)
  xdim, = x.shape
  u = fb(t, x)
  udim, = np.shape(u)

  rhs = lambda t, x: np.reshape(sys(t, x, u), xdim)
  integrator = ode(rhs)
  integrator.set_initial_value(x, tstart)
  integrator.set_integrator('dopri5', **integrator_args)

  result_t = [t]
  result_x = [x.copy()]
  result_u = [u.copy()]
  result_fb_state = []

  if hasattr(fb, 'state'):
    result_fb_state.append(copy(fb.state))

  while t <= tend:
    integrator.integrate(t + step)
    if not integrator.successful():
      print('Warn: integrator doesn\'t feel good')
      break

    x = integrator.y
    t = integrator.t
    u = np.reshape(fb(t, x), udim)

    result_t.append(copy(t))
    result_x.append(copy(x))
    result_u.append(copy(u))

    if hasattr(fb, 'state'):
      result_fb_state.append(copy(fb.state))

  result = SimulationResult(
    t = np.asanyarray(result_t),
    x = np.asanyarray(result_x),
    u = np.asanyarray(result_u),
    state = result_fb_state
  )
  return result
