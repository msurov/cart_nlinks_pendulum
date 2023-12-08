from dataclasses import dataclass

@dataclass
class Rect:
  left : float
  right : float
  bottom : float
  top : float

  @property
  def width(self):
    return self.right - self.left

  @property
  def height(self):
    return self.top - self.bottom

  @property
  def center(self):
    return (self.right + self.left) / 2, (self.top + self.bottom) / 2

def rect_scale(rect : Rect, scale : float) -> Rect:
  w = rect.width * scale
  h = rect.height * scale
  x,y = rect.center
  return Rect(
      left = x - w / 2,
      right = x + w / 2,
      bottom = y - h / 2,
      top = y + h / 2,
  )

def rect_add(rect1 : Rect, rect2 : Rect) -> Rect:
  return Rect(
    rect1.left + rect2.left,
    rect1.right + rect2.right,
    rect1.bottom + rect2.bottom,
    rect1.top + rect2.top,
  )

def covering_rect(rect1 : Rect, rect2 : Rect) -> Rect:
  return Rect(
    left = min(rect1.left, rect2.left),
    right = max(rect1.right, rect2.right),
    bottom = min(rect1.bottom, rect2.bottom),
    top = max(rect1.top, rect2.top),
  )
