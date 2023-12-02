import numpy as np
import matplotlib.pyplot as plt


def solve_quadratic(a, b, c):
  x1 = -b + np.sqrt(b**2 - 4 * a * c) / (a * a)
  x2 = -b - np.sqrt(b**2 - 4 * a * c) / (a * a)
  return x1, x2

def test():
  b = 5.
  c = 3.
  ax1 = []
  ax2 = []
  ax3 = []
  for a in np.linspace(-1, 1, 100):
    x1,x2 = solve_quadratic(a, b, c)
    x3 = -c / b
    ax1.append(x1)
    ax2.append(x2)
    ax3.append(x3)

  plt.plot(ax1)
  plt.plot(ax2)
  plt.plot(ax3)
  plt.show()

test()
