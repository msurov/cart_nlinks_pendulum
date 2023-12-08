from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class MechanicalSystemTrajectory:
  t : np.ndarray
  q : np.ndarray
  dq : np.ndarray
  ddq : np.ndarray | None
  u : np.ndarray | None

@dataclass(frozen=True, slots=True)
class MechanicalSystemPeriodicTrajectory(MechanicalSystemTrajectory):
  period : float
  periodicity_deviation : np.ndarray
