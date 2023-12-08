from ..parameters import DoublePendulumParameters
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import numpy as np
from scipy.interpolate import make_interp_spline
from common.rect import Rect, rect_scale, rect_add


def get_joint_positions(q : np.ndarray, par : DoublePendulumParameters):
  sq = np.sin(q)
  cq = np.cos(q)
  p0 = np.zeros(2)
  p1 = p0 + par.link_length_1 * np.array([sq[0], cq[0]])
  p2 = p1 + par.link_length_2 * np.array([sq[1], cq[1]])
  return np.array([p0, p1, p2])

class DoublePendulumVis:
  def __init__(self, ax, par : DoublePendulumParameters):
    self.par = par
    initial_q = np.zeros(2)
    joint_positions = get_joint_positions(initial_q, self.par)
    joints_x,joints_y = joint_positions.T
    self.chain, = plt.plot(joints_x, joints_y, '-o', linewidth=5, markersize=10, color='royalblue')

  def move(self, q):
    p = get_joint_positions(q, self.par)
    self.chain.set_data(p[:,0], p[:,1])
  
  @property
  def minimal_rect(self):
    w = 0.1 * self.par.link_length_1
    return Rect(-w, w, -w, w)

  @property
  def objects(self):
    return self.chain,

  def set_alpha(self, alpha=1):
    self.chain.set_alpha(alpha)

def estimate_motion_rect(q : np.ndarray, par : DoublePendulumParameters) -> Rect:
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

def animate(t : np.ndarray, q : np.ndarray, par : DoublePendulumParameters, fps=30.0, playspeed=1.0):
  qfun = make_interp_spline(t, q, k=1)

  fig = plt.gcf()
  ax = plt.gca()
  ax.set_aspect(1.0)

  double_pend = DoublePendulumVis(ax, par)
  rect = estimate_motion_rect(q, par)
  rect = rect_add(rect, double_pend.minimal_rect)
  rect = rect_scale(rect, 1.05)
  ax.set_ylim((rect.bottom, rect.top))
  ax.set_xlim((rect.left, rect.right))

  interval = (t[-1] - t[0]) / playspeed
  nframes = int(interval * fps)
  interval = 1000 / fps

  def drawframe(iframe):
    ti = (iframe / fps) * playspeed + t[0]
    double_pend.move(qfun(ti))
    return double_pend.objects

  anim = animation.FuncAnimation(fig, drawframe, frames=nframes, interval=interval, blit=True)
  plt.tight_layout()
  rc('animation', html='jshtml')
  return anim

def draw(q : np.ndarray, par : DoublePendulumParameters):
  assert np.shape(q) == (2,)
  ax = plt.gca()
  ax.set_aspect(1.0)
  double_pend = DoublePendulumVis(ax, par)
  double_pend.move(q)
  rect = estimate_motion_rect(np.array([q]), par)
  rect = rect_add(rect, double_pend.minimal_rect)
  rect = rect_scale(rect, 1.05)
  ax.set_ylim((rect.bottom, rect.top))
  ax.set_xlim((rect.left, rect.right))
  ax.set_ylim((rect.bottom, rect.top))
  ax.set_xlim((rect.left, rect.right))
  return double_pend

def demo_draw():
  par = DoublePendulumParameters(
    link_length_1=1,
    link_length_2=1,
    mass_center_1=0.5,
    mass_center_2=0.5,
    mass_1=1,
    mass_2=1,
    gravity_accel=1
  )
  q = np.array([1., 2])
  pendubot = draw(q, par)
  pendubot.set_alpha(0.5)
  plt.show()

def demo_anim():
  par = DoublePendulumParameters(
    link_length_1=1,
    link_length_2=1,
    mass_center_1=0.5,
    mass_center_2=0.5,
    mass_1=1,
    mass_2=1,
    gravity_accel=1
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
