from systems.cart_pendulum.vis.vis import draw, animate
from parameters.parameters import CartPendParameters
import numpy as np
import matplotlib.pyplot as plt

def test_draw():
  par = CartPendParameters(
    link_lengths = [0.2, 0.3, 0.4],
    mass_center = [0.1, 0.15, 0.2],
    masses = [0.1, 0.1, 0.1, 0.1],
    gravity_accel = 9.81
  )
  q = np.array([0.1, 0.2, -0.3, 0.8])
  fig = plt.figure('pendulum drawing')
  draw(q, par)
  plt.show()

def test_anim():
  par = CartPendParameters(
    link_lengths = [0.2, 0.3, 0.4],
    mass_center = [0.1, 0.15, 0.2],
    masses = [0.1, 0.1, 0.1, 0.1],
    gravity_accel = 9.81
  )
  t = np.arange(0, 5, 0.1)
  initial_q = np.array([1., -2., 3., -4.])
  final_q = np.array([-1., 2., -3., 4.])
  q = initial_q + np.outer(t - t[0], (final_q - initial_q)) / (t[-1] - t[0])

  a = animate(t, q, par)
  plt.show()

if __name__ == '__main__':
  test_draw()
  test_anim()
