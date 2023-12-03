from cas_dynamics.dynamics import (
  CartPendParameters,
  get_cart_pend_dynamics,
  MechanicalSystem
)
from cas_dynamics.mechsys import get_mechsys_normal_form
import casadi as ca
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import make_interp_spline
from motion_planner.trajectory import MechanicalSystemTrajectory
import matplotlib.pyplot as plt
from vis.anim import animate


def find_periodic_traj(theta_fun : ca.SX, period : float, dynamics : MechanicalSystem) -> MechanicalSystemTrajectory:
  B = dynamics.B
  C = dynamics.C
  M = dynamics.M
  G = dynamics.G
  S = ca.DM([
    [0, -1],
    [1, 0]
  ])
  B_perp = B.T @ S

  b1 = (B_perp @ M)[0]
  b2 = (B_perp @ M)[1]
  theta = dynamics.q[1]
  dtheta = dynamics.dq[1]
  ddtheta = ca.SX.sym('ddtheta')

  t = ca.SX.sym('dummy')
  theta_ref = theta_fun(t)
  dtheta_ref = ca.jacobian(theta_ref, t)
  ddtheta_ref = ca.jacobian(dtheta_ref, t)

  b1 = (B_perp @ M)[0]
  b2 = (B_perp @ M)[1]
  ddx_expr = -1/b1 * (b2 * ddtheta + B_perp @ C @ dynamics.dq + B_perp @ G)
  ddx_expr = ca.substitute(
    ddx_expr, 
    ca.vertcat(theta, dtheta, ddtheta, dynamics.dq[0]), 
    ca.vertcat(theta_ref, dtheta_ref, ddtheta_ref, 0),
  )
  ddx_fun = ca.Function('ddx', [t], [ddx_expr])

  def rhs(t, st):
    _, dx = st
    ddx = float(ddx_fun(t))
    return [dx, ddx]

  t_ = np.linspace(0, period, 500)
  sol = solve_ivp(rhs, [0., period], [0., 0.], t_eval=t_, max_step=1e-3)
  assert np.allclose(sol.y[1,-1], 0.), 'Trajectory for x is not periodic'

  x_ = sol.y[0].T
  dx_ = sol.y[1].T
  dx0 = -1 / period * x_[-1]
  x_ += dx0 * t_
  x_[-1] = x_[0]
  dx_ += dx0
  dx_[-1] = dx_[0]

  B_inv = ca.pinv(B)
  u_expr = B_inv @ (M @ ca.vertcat(ddx_expr, ddtheta) + C @ dynamics.dq + G)
  u_expr = ca.substitute(
    u_expr, 
    ca.vertcat(theta, dtheta, ddtheta, dynamics.dq[0]), 
    ca.vertcat(theta_ref, dtheta_ref, ddtheta_ref, 0),
  )
  phase_components_fun = ca.Function(
    'phase_components',
    [t],
    [theta_ref, dtheta_ref, ddtheta_ref, u_expr]
  )
  theta_, dtheta_, ddtheta_, u_ = phase_components_fun(t_)
  ddtheta_ = np.reshape(ddtheta_, (-1,))
  theta_ = np.reshape(theta_, (-1,))
  dtheta_ = np.reshape(dtheta_, (-1,))
  u_ = np.reshape(u_, (-1,))
  u_[-1] = u_[0]
  theta_[-1] = theta_[0]
  dtheta_[-1] = dtheta_[0]
  ddtheta_[-1] = ddtheta_[0]

  ddx_ = ddx_fun(t_)
  ddx_ = np.reshape(ddx_, (-1,))
  ddx_[-1] = ddx_[0]

  return MechanicalSystemTrajectory(
      t = t_, 
      q = np.array([x_, theta_]).T,
      dq = np.array([dx_, dtheta_]).T,
      ddq = np.array([ddx_, ddtheta_]).T,
      u = u_
  )

def show_trajectory(traj : MechanicalSystemTrajectory):
  fig, axes = plt.subplots(2, 2, sharex=True, num='phase trajectory components')
  plt.sca(axes[0,0])
  plt.grid(True)
  plt.plot(traj.q[:,1], traj.q[:,0])
  plt.ylabel(R'$x$')
  plt.sca(axes[0,1])
  plt.grid(True)
  plt.plot(traj.q[:,1], traj.dq[:,1])
  plt.ylabel(R'$\dot\theta$')
  plt.sca(axes[1,0])
  plt.grid(True)
  plt.xlabel(R'$\theta$')
  plt.plot(traj.q[:,1], traj.dq[:,0])
  plt.ylabel(R'$\dot x$')
  plt.sca(axes[1,1])
  plt.grid(True)
  plt.plot(traj.q[:,1], traj.u)
  plt.ylabel(R'$u$')
  plt.xlabel(R'$\theta$')
  plt.tight_layout()
  return fig

def verify_trajectory(mechsys : MechanicalSystem, traj : MechanicalSystemTrajectory):
  qdim = mechsys.qdim
  u_fun = make_interp_spline(traj.t, traj.u)
  sys = get_mechsys_normal_form(mechsys)

  def rhs(t, x):
    return sys(t, x, u_fun(t))

  plt.figure('trajectory verification')
  plt.subplot(311)
  plt.gca().set_prop_cycle(None)
  plt.ylabel(R'$q$')
  plt.grid(True)
  plt.plot(traj.t, traj.q, '--', lw=2)

  plt.subplot(312)
  plt.gca().set_prop_cycle(None)
  plt.ylabel(R'$\dot q$')
  plt.grid(True)
  plt.plot(traj.t, traj.dq, '--', lw=2)

  plt.subplot(313)
  plt.gca().set_prop_cycle(None)
  plt.ylabel(R'$\ddot q$')
  plt.grid(True)
  plt.plot(traj.t, traj.ddq, '--', lw=2)

  npts = 100
  for i in np.arange(0, traj.t.shape[0], npts):
    t = traj.t[i:i+npts]
    q0 = traj.q[i]
    dq0 = traj.dq[i]
    x0 = np.concatenate((q0, dq0))
    sol = solve_ivp(rhs, [t[0], t[-1]], x0, t_eval=t, max_step=1e-3, atol=1e-12, rtol=1e-12)

    ok = np.allclose(sol.y[0:qdim].T, traj.q[i:i+npts], atol=1e-3, rtol=1e-3)
    plt.subplot(311)
    if ok:
      plt.gca().set_prop_cycle(None)
    else:
      print('[warn] trajectory is not correct')
      plt.gca().set_prop_cycle(color=['red'])
    plt.plot(sol.t, sol.y[0:qdim].T, '-', lw=1)

    ok = np.allclose(sol.y[qdim:2*qdim].T, traj.dq[i:i+npts], atol=1e-3, rtol=1e-3)
    plt.subplot(312)
    if ok:
      plt.gca().set_prop_cycle(None)
    else:
      print('[warn] trajectory is not correct')
      plt.gca().set_prop_cycle(color=['red'])
    plt.plot(sol.t, sol.y[qdim:2*qdim].T, '-', lw=1)

    ddq = np.array([rhs(t, x)[qdim:2*qdim] for t,x in zip(sol.t, sol.y.T)])
    plt.subplot(313)
    if ok:
      plt.gca().set_prop_cycle(None)
    else:
      print('[warn] trajectory is not correct')
      plt.gca().set_prop_cycle(color=['red'])
    plt.plot(sol.t, ddq, '-', lw=1)

def demo():
  par = CartPendParameters(
    link_lengths=[2],
    mass_center=[1],
    masses=[1, 1],
    gravity_accel=1
  )
  mechsys = get_cart_pend_dynamics(par)
  t = ca.SX.sym('t')
  # theta_ref = ca.Function('theta_ref', [t], [0.11 * ca.sin(t) - 0.26 * ca.sin(2 * t)])
  # theta_ref = ca.Function('theta_ref', [t], [0.14 * ca.sin(t) + 0.19 * ca.sin(3*t) - 0.11 * ca.sin(5*t)])
  theta_ref = ca.Function('theta_ref', [t], [0.6 * ca.sin(t) - 0.4 * ca.sin(2*t)])
  traj = find_periodic_traj(theta_ref, 2 * np.pi, mechsys)
  a = animate(traj.t, traj.q, par)
  a.save('fig/wierd_trajectory.gif')
  # verify_trajectory(mechsys, traj)
  # show_trajectory(traj)
  # plt.show()

if __name__ == '__main__':
  demo()
