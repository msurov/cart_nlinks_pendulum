import numpy as np
from scipy.integrate import ode, solve_ivp
from dataclasses import dataclass
from typing import Callable


@dataclass(slots=True)
class ODESolution:
  t : np.ndarray  # solution knots
  x : np.ndarray  # values of the found function at the knots

@dataclass(slots=True)
class ODEPeriodicSolution(ODESolution):
  period : float
  periodicity_deviation : np.ndarray


def segment_point_distance(seg : tuple[np.ndarray], pt : np.ndarray) -> tuple[float, float]:
  p1, p2 = seg
  p21 = p2 - p1
  t = np.dot(pt - p1, p21) / np.dot(p21, p21)
  t = np.clip(t, 0, 1)
  dist = np.linalg.norm(p21 * t + p1 - pt)
  return dist, t

def solve_quadratic(a, b, c):
  d = b**2 - 4 * a * c
  if d < 0:
    r1 = 0.5 * (-b + 1.j * np.sqrt(-d)) / a
    r2 = 0.5 * (-b - 1.j * np.sqrt(-d)) / a
    return r1, r2
  elif d == 0:
    r1 = -0.5 * b / a
    return r1
  else:
    r1 = 0.5 * (-b + np.sqrt(d)) / a
    r2 = 0.5 * (-b - np.sqrt(d)) / a
    return r1, r2

def solve_periodic_ivp(
    sys : Callable[[float, np.ndarray], np.ndarray],
    tstart : float,
    xstart : np.ndarray,
    step : float,
    period_max=None,
    eps=1e-3,
    **integrator_args
  ) -> ODEPeriodicSolution:
  R"""
  Description:
    Find periodic solution of the initial value problem for the given ODE and initial conditons. 
    It is assumed that the given initial value correspond to a periodic solution, otherwise the function will fail.

  Arguments:
    sys: right hand side of the ODE
    tstart, xstart: are the initial time and initial state of the IVP
    step: is the step to iterate the solution
    period_max: if not None then this value defines the maximum possible period
    eps: if trajectory comes back to the original point closer than eps, the solver assumes that the trajectory is closed

  Return:
    Solution of the IVP or None
  """
  t = tstart
  xstart = np.array(xstart)
  x = np.copy(xstart)

  integrator = ode(sys)
  integrator.set_initial_value(xstart, t)
  integrator.set_integrator('dopri5', **integrator_args)

  result_t = [t]
  result_x = [x.copy()]

  while True:
    if period_max is not None:
      assert t <= period_max, f"The trajectory is longer than {period_max} or not periodic"

    integrator.integrate(t + step)
    if not integrator.successful():
      errcode = integrator.get_return_code()
      assert errcode != -1, f"Integrator failed: Input is not consistent"

    x_new = integrator.y
    t_new = integrator.t

    seg = (x, x_new)
    dist, l = segment_point_distance(seg, xstart)
    if l > 0 and dist < eps and len(result_t) > 1:
      break

    t = t_new
    x = np.copy(x_new)
    result_t.append(t)
    result_x.append(x)

  fstart = sys(tstart, xstart)
  f1 = sys(t, x)
  f2 = sys(t_new, x_new)

  tend = t_new - np.dot(fstart, x_new - xstart) / np.dot(fstart, fstart)
  integrator.integrate(tend)
  xend = integrator.y
  periodicity_deviation = xstart - xend
  assert np.linalg.norm(periodicity_deviation) < eps, "Computations are wrong. Who knows why.."

  seg = (x, xend)
  dist, l = segment_point_distance(seg, xstart)

  result_t.append(tend)
  result_x.append(np.copy(xstart))
  t = np.array(result_t)
  x = np.array(result_x)

  return ODEPeriodicSolution(
    t = t,
    x = x,
    period = tend - tstart,
    periodicity_deviation = periodicity_deviation
  )

def integrate_twice(
    f : Callable[[float], float],
    t : np.ndarray,
    **integrator_args
  ) -> np.ndarray:
  """
  Integrate the given function twice at the given knots
  """
  def sys(t, x):
    return np.array([x[1], f(t)])
  sol = solve_ivp(sys, [t[0], t[-1]], [0., 0.], t_eval=t, **integrator_args)
  return sol.y[0], sol.y[1]
