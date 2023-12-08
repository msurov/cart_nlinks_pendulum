from dataclasses import dataclass
from numbers import Number


@dataclass(frozen=True, slots=True)
class CartPendParameters:
  """
    Physical partameters of the Cart N-Links Pendulum system
  """
  link_lengths : list[Number]
  mass_center : list[Number]
  masses : list[Number]
  gravity_accel : Number

  def __post_init__(self):
    assert len(self.mass_center) == self.nlinks
    assert len(self.masses) == self.nlinks + 1

  @property
  def nlinks(self):
    return len(self.link_lengths)
