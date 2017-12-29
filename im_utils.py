import scipy.misc as scm
import PIL
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
  #Convert to PIL.Image format
  im = scm.toimage(im)
  interpolation = {}
  interpolation['bilinear'] = Image.BILINEAR
  if interp not in interpolation:
    raise KeyError('Interpolation of type {} is invalid'.format(interp))
  imr = im.rotate(angle, interpolation[interp], expand=expand)
  return scm.fromimage(imr)