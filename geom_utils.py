import math
import numpy as np

class Point(object):
  def __init__(self, x, y):
    self._x = x
    self._y = y
    
  @property
  def x(self):
    return self._x
  
  @property
  def y(self):
    return self._y
    
  @property
  def x_int(self):
    return int(np.round(self.x))
    
  @property
  def y_int(self):
    return int(np.round(self.y))
    
  def rotate(self, theta):
    """
     Rotate by angle theta (in degrees) in counter-clockwise manner
    """
    x, y = self.x, self.y
    theta_current = math.atan2(y, x)
    theta_new     = theta_current + math.radians(theta)
    r = np.sqrt(x*x + y*y)
    self._x = r * math.cos(theta_new)
    self._y = r * math.sin(theta_new)
    

def rotate_point(x, y, theta):
  """
  Rotate (x,y) by angle theta (in degrees) in counter-clockwise manner
  """
  pt = Point(x, y)
  pt.rotate(theta)
  return pt.x_int, pt.y_int

def rotate_box(x1, y1, x2, y2, theta):
  x1, y1 = rotate_point(x1, y1, theta)
  x2, y2 = rotate_point(x2, y2, theta)
  #reassing points which is top-left and bottom-right
  if x1 > x2:
    _x1 = x2
    x2 = x1
    x1 = _x1
  if y1 > y2:
    _y1 = y2
    y2  = y1
    y1  = _y1
  return x1, y1, x2, y2