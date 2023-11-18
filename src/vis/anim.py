from sym_dynamics.dynamics import CartPendParameters, link_positions
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML
import numpy as np
from scipy.interpolate import make_interp_spline
from dataclasses import dataclass
from matplotlib.patches import Arc, Ellipse, Circle
from common.rect import Rect, rect_scale


def get_joint_positions(q : np.ndarray, par : CartPendParameters):
  x = q[0]
  thetas = q[1:]
  sin_theta = np.sin(thetas)
  cos_theta = np.cos(thetas)
  links_lengths = np.array(par.link_lengths)
  joints_positions = np.zeros((par.nlinks + 1, 2))
  joints_positions[0,0] = x
  joints_positions[1:,0] = sin_theta * links_lengths
  joints_positions[1:,1] = cos_theta * links_lengths
  joints_positions = np.cumsum(joints_positions, axis=0)
  return joints_positions

class CartPendVis:
  def __init__(self, par : CartPendParameters):
    self.par = par
    initial_q = np.zeros(par.nlinks + 1)
    joint_positions = get_joint_positions(initial_q, self.par)
    joints_x,joints_y = joint_positions.T
    self.chain, = plt.plot(joints_x, joints_y, '-o', linewidth=5, markersize=10)

  def move(self, q):
    p = get_joint_positions(q, self.par)
    self.chain.set_data(p[:,0], p[:,1])

  def elems(self):
    return self.chain,

def estimate_motion_rect(q : np.ndarray, par : CartPendParameters) -> Rect:
  p = get_joint_positions(q[0,:], par)
  left,bottom = np.min(p, axis=0)
  right,top = np.max(p, axis=0)
  for qi in q[1:]:
    p = get_joint_positions(qi, par)
    xmin,ymin = np.min(p, axis=0)
    xmax,ymax = np.max(p, axis=0)
    left = min(left, xmin)
    right = max(right, xmax)
    bottom = min(bottom, ymin)
    top = max(top, ymax)
  return Rect(left,right,bottom,top)

def animate(t : np.ndarray, q : np.ndarray, par : CartPendParameters, fps=30.0):
  qfun = make_interp_spline(t, q, k=1)

  fig = plt.gcf()
  ax = plt.gca()
  ax.set_aspect(1.0)
  plt.grid(True)

  cart_pend = CartPendVis(par)
  rect = estimate_motion_rect(q, par)
  rect = rect_scale(rect, 1.05)
  ax.set_ylim((rect.bottom, rect.top))
  ax.set_xlim((rect.left, rect.right))

  interval = t[-1] - t[0]
  nframes = int(interval * fps)
  interval = 1000 / fps

  def drawframe(iframe):
    ti = iframe / fps + t[0]
    cart_pend.move(qfun(ti))
    return cart_pend.elems()

  anim = animation.FuncAnimation(fig, drawframe, frames=nframes, interval=interval, blit=True)
  plt.tight_layout()
  rc('animation', html='jshtml')
  return anim

def draw(q : np.ndarray, par : CartPendParameters):
  assert np.shape(q) == (par.nlinks + 1,)

  cart_pend = CartPendVis(par)
  cart_pend.move(q)
  q = np.reshape(q, (1, -1))
  rect = estimate_motion_rect(q, par)
  rect = rect_scale(rect, 1.05)
  ax = plt.gca()
  plt.axis('equal')
  plt.grid(True)
  ax.set_ylim((rect.bottom, rect.top))
  ax.set_xlim((rect.left, rect.right))
  return cart_pend.elems()
