import math
import numpy as np
from pyhelper_fns import vis_utils, im_utils

class Point(object):
  def __init__(self, x, y, imH=None):
    """
      x, y: x, y coordinates of a point
      imH: height of the image if the point is in image
    """
    self._x = x
    self._y = y
    self._imH = imH
    
  @classmethod
  def from_imcoords(cls, x, y, imH):
    """ if x, y are expressed in image coordinates"""
    self = cls(x, imH-y, imH)
    return self
    
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
    
  @property
  def image_x(self):
    """ in image coordinate frame"""
    return self.x
    
  @property
  def image_y(self):
    """ in image coordinate frame"""
    assert self._imH is not None
    return self._imH - self.y
    
  @property
  def image_x_int(self):
    return int(np.round(self.image_x))
    
  @property
  def image_y_int(self):
    return int(np.round(self.image_y))
    
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
    

def rotate_point(x, y, theta, imCoords=False, imH=None, isFloat=False):
  """
  Rotate (x,y) by angle theta (in degrees) in counter-clockwise manner
  """
  if imCoords:
    pt = Point.from_imcoords(x, y, imH)
  else:
    pt = Point(x, y)
  pt.rotate(theta)
  if imCoords:
    return pt.image_x_int, pt.image_y_int
  else:
    return pt.x_int, pt.y_int
  

def rotate_box(x1, y1, x2, y2, theta, **kwargs):
  """
  x1, y1: top left
  x2, y2: bottom right
  """
  x1, y1 = rotate_point(x1, y1, theta, **kwargs)
  x2, y2 = rotate_point(x2, y2, theta, **kwargs)
  #reassing points which is top-left and bottom-right
  if x1 > x2:
    _x1 = x2
    x2 = x1
    x1 = _x1
  if 'imCoords' in kwargs and kwargs['imCoords']:
    if y1 > y2:
      _y1 = y2
      y2  = y1
      y1  = _y1
  else:
    if y2 > y1:
      _y1 = y2
      y2  = y1
      y1  = _y1
  return x1, y1, x2, y2
  
def rotate_box_around_center(x1, y1, x2, y2, theta, center, **kwargs):
  """
  rotate the box around the center
  """
  cx, cy = center
  cxRot, cyRot = rotate_point(cx, cy, theta)
  cxRot, cyRot = np.abs(cxRot), np.abs(cyRot)
  x1, x2 = x1 - cx, x2 - cx
  if 'imCoords' in kwargs and kwargs['imCoords']:
    kwargs['imH'] = 2 * cyRot
    imH = 2 * cy
    _y1, _y2 = imH - y1, imH - y2
    y1, y2 = _y1 - cy, _y2 - cy
  else:
    y1, y2 = y1 - cy, y2 - cy
  #Rotate the box now, coordinates are wrt to origin (not top-left of image)
  x1, y1, x2, y2 = rotate_box(x1, y1, x2, y2, theta, imCoords=False, imH=None)
  #Recenter
  x1, x2 = x1 + cxRot, x2 + cxRot
  if 'imCoords' in kwargs and kwargs['imCoords']:
    imH = 2 * cyRot
    y1,  y2 = imH - (y1 + cyRot), imH - (y2 + cyRot) 
  else:
    y1, y2 = y1 + cyRot, y2 + cyRot
  return x1, y1, x2, y2
  
def get_test(rot=-90):
  if rot == [-90, 90]:
    imH, imW = 524, 640
    bboxs = [[248.093353495, 310.126402835, 363.097447045, 447.094726891],
          [404.6338720575185, 3.8798152359500024, 470.34534383500056, 164.1069113020001],
          [518.153754212, 209.335037559, 558.211359831, 290.740739593],
          [258.430853758, 33.6058491098, 544.002816393, 510.410675305],
          [447.085525616, 0, 538.830364291, 161.522549649]]
          
  else:
    imH, imW = 524, 640
    bboxs = [[260, 236, 378, 432],
             [181, 1, 281, 110],
             [90, 451, 344, 523]]
  return imH, imW, bboxs
  
  
def test_rot_bbox_im(rot=-90):
  imH, imW, bboxs = get_test(rot)
  center = (imW / 2, imH / 2)
  im = np.zeros((imH, imW,3), dtype=np.uint8)
  subp = vis_utils.Subplot()
  for b in bboxs:
    im1 = vis_utils.draw_square_on_im(np.copy(im), b, col='blue')
    imr = im_utils.imrotate(im1, rot)
    #Rotate the box
    boxRot = rotate_box_around_center(b[0], b[1], b[2], b[3], rot, center,
              imCoords=True, imH=imH)
    print (b)
    print (boxRot)
    print (b[2] - b[0], b[3] - b[1])
    print (boxRot[2] - boxRot[0], boxRot[3] - boxRot[1])
    im2 = vis_utils.draw_square_on_im(imr, boxRot, col='red')
    subp.plot_image(im2)
    ip = raw_input()
    if ip =='q':
      return
    
    