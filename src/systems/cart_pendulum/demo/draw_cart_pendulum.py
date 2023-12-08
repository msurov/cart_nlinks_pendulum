from ..cas.dynamics import CartPendParameters
from ..vis import draw
import matplotlib.pyplot as plt
import numpy as np


def cart_pend_schematic():
  par = CartPendParameters(
    link_lengths=[1.],
    mass_center=[1.],
    masses=[0.1, 0.1],
    gravity_accel=1
  )
  fig = plt.figure("horizontal oscillations schematic")
  plt.axvline(0, ls='--', color='grey')
  plt.axhline(-0.18, lw=7, ls='-', color='black')
  ax = plt.gca()
  q = np.array([0., 0.8])
  draw(q, par)

  an1 = ax.annotate(
    "",
    xy=(0.5*np.sin(q[1]), 0.5*np.cos(q[1])),
    xycoords='data',
    xytext=(0.5*np.sin(0), 0.5*np.cos(0)),
    textcoords='data',
    size=24,
    arrowprops={
      'arrowstyle': "-|>",
      'connectionstyle': "arc3,rad=-0.2",
      'relpos': (0., 0.),
      'shrinkA': 2,
      'shrinkB': 5
    }
  )
  an2 = ax.annotate(
    R"$\phi$",
    xy=(0,0),
    xytext=(0.55*np.sin(q[1]/2), 0.55*np.cos(q[1]/2)),
    textcoords='data',
    size=28,
    va="center",
    ha="left",
  )
  an3 = ax.annotate(
    "",
    xy=(0.6, 0),
    xycoords='data',
    xytext=(0.25, 0),
    textcoords='data',
    size=24,
    arrowprops={
      'arrowstyle': "-|>",
      'relpos': (0., 0.),
      'shrinkA': 0,
      'shrinkB': 5
    }
  )
  an4 = ax.annotate(
    R"$f$",
    xy=(0.6, 0),
    xycoords='data',
    xytext=(0.5, 0.05),
    textcoords='data',
    size=28,
  )
  an5 = ax.annotate(
    R"$x$",
    xy=(0., 0.),
    xycoords='data',
    xytext=(0.0, -0.25),
    va="center",
    ha="center",
    textcoords='data',
    size=28,
  )
  ax.set_xticks([0], [])
  ax.set_yticks([0], [])
  fig.tight_layout()
  fig.savefig('fig/cart_pendulum_schematic.png', transparent=True)

def cart_pend_horiz_position():  
  par = CartPendParameters(
    link_lengths=[1.],
    mass_center=[1.],
    masses=[0.1, 0.1],
    gravity_accel=1
  )
  q0 = np.array([0., 2.])
  fig = plt.figure("horizontal position schematic")
  ax = plt.gca()
  draw(q0, par)
  fig.savefig('fig/horizontal_position.png', transparent=True)

def main():
  plt.rcParams.update({
      "text.usetex": True,
      "font.family": "Helvetica",
      'font.size': 16
  })
  cart_pend_schematic()
  cart_pend_horiz_position()
