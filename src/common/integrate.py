import numpy as np
from scipy.integrate import ode


def solve_periodic_ivp(sys, period_max, step, xstart, **integrator_args):

  t = 0.
  x = np.array(xstart)
  xdim, = x.shape

  rhs = lambda t, x: np.reshape(sys(t, x, u), xdim)
  integrator = ode(rhs)
  integrator.set_initial_value(xstart, tstart)
  integrator.set_integrator('dopri5', **integrator_args)

  result_t = [t]
  result_x = [x.copy()]

  while t <= period_max:
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