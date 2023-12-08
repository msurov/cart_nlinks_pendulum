from ..cas.dynamics import CartPendParameters, link_positions
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML
import numpy as np
from scipy.interpolate import make_interp_spline
from dataclasses import dataclass
from matplotlib.patches import Arc, Ellipse, Circle, Polygon
from common.rect import Rect, rect_scale, covering_rect, rect_add


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

class CartVis:
  def __init__(self, scale):
    self.scale = scale
    self.xy = np.array([0, 0])
    pts = self.get_verts()
    self.poly = Polygon(pts, closed=True, fill=True, 
                        facecolor='lightsalmon', edgecolor='dimgrey', linewidth=0.5,
                        linestyle='-')
    p1, p2 = self.get_wheels_positions()
    self.wheel1 = Circle(p1, self.wheel_radius, fill=True, 
                         facecolor='grey', edgecolor='black', linewidth=0.5,
                         linestyle='-')
    self.wheel2 = Circle(p2, self.wheel_radius, fill=True, 
                         facecolor='grey', edgecolor='black', linewidth=0.5,
                         linestyle='-')

  def get_wheels_positions(self):
    p1 = np.array([-1/3, -1/5]) * self.scale + self.xy
    p2 = np.array([1/3, -1/5]) * self.scale + self.xy
    return p1, p2

  def get_verts(self):
    pts = np.array([
      [-1/2, -1/4],
      [-1/2, 0],
      [-1/3, 1/6],
      [1/3, 1/6],
      [1/2, 0],
      [1/2, -1/4],
    ])
    return pts * self.scale + self.xy
  
  @property
  def wheel_radius(self):
    return self.scale / 8

  @property
  def objects(self):
    return self.poly, self.wheel1, self.wheel2
  
  def move(self, x, y):
    self.xy = np.array([x, y])
    pts = self.get_verts()
    self.poly.set_xy(pts)
    p1, p2 = self.get_wheels_positions()
    self.wheel1.set(center=p1)
    self.wheel2.set(center=p2)
  
  @property
  def coverging_rect(self):
    pts = self.get_verts()
    p1, p2 = self.get_wheels_positions()
    r = self.wheel_radius
    rect1 = Rect(p1[0] - r, p1[0] + r, p1[1] - r, p1[1] + r)
    rect2 = Rect(p2[0] - r, p2[0] + r, p2[1] - r, p2[1] + r)
    rect3 = Rect(
      np.min(pts[:,0]),
      np.max(pts[:,0]),
      np.min(pts[:,1]),
      np.max(pts[:,1]),
    )
    return covering_rect(rect1, covering_rect(rect2, rect3))

class CartPendVis:
  def __init__(self, ax, par : CartPendParameters):
    self.par = par
    initial_q = np.zeros(par.nlinks + 1)
    joint_positions = get_joint_positions(initial_q, self.par)
    joints_x,joints_y = joint_positions.T
    self.chain, = plt.plot(joints_x, joints_y, '-o', linewidth=5, markersize=10, color='royalblue')
    scale = par.link_lengths[0] / 2
    self.cart = CartVis(scale)
    [ax.add_patch(obj) for obj in self.cart.objects]

  def move(self, q):
    p = get_joint_positions(q, self.par)
    self.chain.set_data(p[:,0], p[:,1])
    self.cart.move(q[0], 0)
  
  @property
  def minimal_rect(self):
    return self.cart.coverging_rect

  @property
  def objects(self):
    return self.cart.objects + (self.chain,)

  def set_alpha(self, alpha=1):
    for obj in self.cart.objects:
      obj.set_alpha(alpha)
    self.chain.set_alpha(alpha)

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

def animate(t : np.ndarray, q : np.ndarray, par : CartPendParameters, fps=30.0, playspeed=1.0):
  qfun = make_interp_spline(t, q, k=1)

  fig = plt.gcf()
  ax = plt.gca()
  ax.set_aspect(1.0)

  cart_pend = CartPendVis(ax, par)
  rect = estimate_motion_rect(q, par)
  rect = rect_add(rect, cart_pend.minimal_rect)
  rect = rect_scale(rect, 1.05)
  ax.set_ylim((rect.bottom, rect.top))
  ax.set_xlim((rect.left, rect.right))

  interval = (t[-1] - t[0]) / playspeed
  nframes = int(interval * fps)
  interval = 1000 / fps

  def drawframe(iframe):
    ti = (iframe / fps) * playspeed + t[0]
    cart_pend.move(qfun(ti))
    return cart_pend.objects

  anim = animation.FuncAnimation(fig, drawframe, frames=nframes, interval=interval, blit=True)
  plt.tight_layout()
  rc('animation', html='jshtml')
  return anim

def draw(q : np.ndarray, par : CartPendParameters):
  assert np.shape(q) == (par.nlinks + 1,)
  ax = plt.gca()
  ax.set_aspect(1.0)
  cart_pend = CartPendVis(ax, par)
  cart_pend.move(q)
  rect = estimate_motion_rect(np.array([q]), par)
  rect = rect_add(rect, cart_pend.minimal_rect)
  rect = rect_scale(rect, 1.05)
  ax.set_ylim((rect.bottom, rect.top))
  ax.set_xlim((rect.left, rect.right))
  ax.set_ylim((rect.bottom, rect.top))
  ax.set_xlim((rect.left, rect.right))
  return cart_pend

def demo_draw():
  par = CartPendParameters(
    link_lengths=[1., 1.],
    mass_center=[0.5, 0.5],
    masses=[1., 1., 1.],
    gravity_accel=10
  )
  q = np.array([1., 2, 3.])
  cartpend = draw(q, par)
  cartpend.set_alpha(0.5)
  plt.show()

def demo_anim():
  par = CartPendParameters(
    link_lengths=[1., 1.],
    mass_center=[0.5, 0.5],
    masses=[1., 1., 1.],
    gravity_accel=10
  )
  t = np.linspace(0, 5, 100)
  q = np.array([
    np.sin(t),
    np.sin(2*t),
    np.sin(3*t),
  ]).T
  a = animate(t, q, par)
  plt.show()

if __name__ == '__main__':
  demo_draw()
  demo_anim()
