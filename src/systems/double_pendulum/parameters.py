from dataclasses import dataclass
from numbers import Number

@dataclass
class DoublePendulumParameters:
  mass_1 : [Number]
  mass_2 : [Number]
  mass_center_1 : [Number]
  mass_center_2 : [Number]
  link_length_1 : [Number]
  link_length_2 : [Number]
  gravity_accel : [Number]
  actuated_joint : int
