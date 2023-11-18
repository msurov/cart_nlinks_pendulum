from time import time

class PerfCounter:
  def __init__(self):
    self.t = None
    self.name = None

  def __call__(self, name : str):
    t = time()
    if self.t is None:
      self.t = t
      self.name = name
    else:
      print(f'{name} done in {t - self.t:.6f}sec')
      self.t = t
      self.name = name
