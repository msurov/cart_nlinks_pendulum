from dataclasses import dataclass
import sympy as sy


@dataclass(frozen=True, slots=True)
class SymFunction:
  expr : sy.Tuple
  arg : sy.Symbol
