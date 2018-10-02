import numpy as np
from PIL import Image

def imrotate(im, angle, interp='bilinear', expand=True):
  """
  rotates the image
  Args:
    im: np.ndarray
    angle:degrees by which the image should be rotated counter-clockwise
    interp: kind of interpolation
    expand: refer to PIL.Image.rotate(), whether to expand the image size to
            match the size after rotation or not
  Returns:
    np.array of rotated image
  """
  im = Image.fromarray(im[:, :, [2, 1, 0]], 'RGB')
  interpolation = {}
  interpolation['bilinear'] = Image.BILINEAR
  if interp not in interpolation:
    raise KeyError('Interpolation of type {} is invalid'.format(interp))
  imr = im.rotate(angle, interpolation[interp], expand=expand)
  return np.array(im)[:, :, [2, 1, 0]]
