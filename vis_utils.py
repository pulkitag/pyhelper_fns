## @package vis_utils
#  Miscellaneous Functions for visualizations
#
from math import sqrt, ceil
import numpy as np
import scipy.misc as scm
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import copy
import os
from os import path as osp
import pdb, time, shutil
from matplotlib import gridspec
from functools import reduce
from pyhelper_fns import path_utils

class Subplot(object):
  """
  In many cases just a handle to axis is required
  """
  def __init__(self):
    self.fig = plt.figure()
    self.ax = self.fig.add_subplot(111)
    plt.ion()
    
  def plot_image(self, im, title=None):
    self.ax.imshow(im)
    if title is not None:
      self.fig.canvas.set_window_title(title)
    plt.show()


class SubplotMulti(object):
  """
  Create multiple place-holder axes
  """
  def __init__(self, numPlots=2, subPlotShape=None, axTitles=None):
    #determine the shape in which plots are to be made   
    plt.ion()
    if subPlotShape is None:
      assert numPlots is not None
      N = np.ceil(np.sqrt(numPlots)).astype(np.int)
      subPlotShape = (N,N)
    else:
      numPlots = reduce(lambda x, y: x*y, subPlotShape)
    #default tiltes of axes
    if axTitles is None:
      axTitles = ['plot %d' % (i+1) for i in range(numPlots)]
    else:
      assert len(axTitles) == numPlots, len(axTitles)
    #prepare the figure
    self.fig        = plt.figure()
    self.axs        = []
    self.numPlots   = numPlots 
    self.axTitles   = axTitles
    for i in range(numPlots):
      shp = subPlotShape + (i+1,)
      aa  = self.fig.add_subplot(*shp)
      self.axs.append(aa)
      if axTitles is not None:
        self.axs[i].set_title(axTitles[i])
        self.axs[i].autoscale(True)
        
  def plot_image(self, ims, title=None):
    assert type(ims) in [list, tuple], 'ims is not in correct format'
    for i, im in enumerate(ims):
      self.axs[i].imshow(im)
    if title is not None:
      self.fig.canvas.set_window_title(title)
    plt.show()


##
#Plot n images
def plot_n_ims(ims, fig=None, titleStr='', figTitle='',
				 axTitles = None, subPlotShape=None,
				 isBlobFormat=False, chSwap=None, trOrder=None,
         showType=None):
  '''
    ims: list of images
    isBlobFormat: Caffe stores images as ch x h x w
                  True - convert the images into h x w x ch format
    trOrder     : If certain transpose order of channels is to be used
                  overrides isBlobFormat
    showType    : imshow or matshow (by default imshow)
  '''
  ims = copy.deepcopy(ims)
  if trOrder is not None:
    for i, im in enumerate(ims):
      ims[i] = im.transpose(trOrder)
  if trOrder is None and isBlobFormat:
    for i, im in enumerate(ims):
      ims[i] = im.transpose((1,2,0))
  if chSwap is not None:
    for i, im in enumerate(ims):
      ims[i] = im[:,:,chSwap]
  plt.ion()
  if fig is None:
    fig = plt.figure()
  plt.figure(fig.number)
  plt.clf()
  if subPlotShape is None:
    N = np.ceil(np.sqrt(len(ims)))
    subPlotShape = (N,N)
    #gs = gridspec.GridSpec(N, N)
  ax = []
  for i in range(len(ims)):
    shp = subPlotShape + (i+1,)
    aa  = fig.add_subplot(*shp)
    aa.autoscale(False)
    ax.append(aa)
    #ax.append(plt.subplot(gs[i]))

  if showType is None:
    showType = ['imshow'] * len(ims)
  else:
    assert len(showType) == len(ims)

  for i, im in enumerate(ims):
    ax[i].set_ylim(im.shape[0], 0)
    ax[i].set_xlim(0, im.shape[1])
    if showType[i] == 'imshow':
      ax[i].imshow(im.astype(np.uint8))
    elif showType[i] == 'matshow':
      res = ax[i].matshow(im)
      plt.colorbar(res, ax=ax[i])
    ax[i].axis('off')
    if axTitles is not None:
      ax[i].set_title(axTitles[i])
  if len(figTitle) > 0:
    fig.suptitle(figTitle)
  plt.show()
  return ax


def plot_pairs(im1, im2, **kwargs):
	ims = []
	ims.append(im1)
	ims.append(im2)
	return plot_n_ims(ims, subPlotShape=(1,2), **kwargs)
	

##
#Plot pairs of images from an iterator_fun
def plot_pairs_iterfun(ifun, **kwargs):
	'''
		ifun  : iteration function
		kwargs: look at input arguments for plot_pairs
	'''
	plt.ion()
	fig = plt.figure()
	pltFlag = True
	while pltFlag:
		im1, im2 = ifun()
		plot_pairs(im1, im2, fig=fig, **kwargs)
		ip = raw_input('Press Enter for next pair')
		if ip == 'q':
			pltFlag = False	


class MyAnimation(object):
  def __init__(self, vis_func, frames=100, fps=20, height=200, width=200, fargs=[]):
    self.frames = frames
    self.vis_func = vis_func
    self.vis_func_args = fargs
    self.fps = fps
    self.fig, self.ax = plt.subplots(1,1)
    plt.show(block=False)
    self.bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)
    im = np.zeros((height, width, 3)).astype(np.uint8)
    self.image_obj = self.ax.imshow(im)
    self.fig.canvas.draw()

  def __del__(self):
    plt.close(self.fig)

  def run(self, fargs=[]):
    if len(fargs)==0:
      func_args = self.vis_func_args
    else:
      func_args = fargs
    time_diff = float(1.0)/self.fps
    for i in range(self.frames):
      op = self.vis_func(i, *func_args)
      if type(op) == tuple:
        im, is_stop = op
      else:
        im = op
        is_stop = False
      self._display(im)
      time.sleep(time_diff)
      if is_stop:
        break
      
  def _display(self, pixels, deltaT=1e-3):
    self.image_obj.set_data(pixels)
    self.fig.canvas.restore_region(self.bg)
    self.ax.draw_artist(self.image_obj)
    self.fig.canvas.blit(self.ax.bbox)
    plt.pause(deltaT)


class MyAnimationMulti(object):
  """Animate mulitple subplots simultaneously"""
  def __init__(self, vis_func, height=200, width=200, 
               subPlotShape=None, numPlots=None,
               isIm=None, axTitles=None):
    """
    Args:
      isIm: which of plots are going to be images
    """
    self.vis_func = vis_func
    if subPlotShape is None:
      assert numPlots is not None
      N = np.ceil(np.sqrt(numPlots)).astype(np.int)
      subPlotShape = (N,N)
    else:
      numPlots = reduce(lambda x, y: x*y, subPlotShape)
    if isIm is None:
      isIm = numPlots * [True]
    if axTitles is None:
      axTitles = ['plot %d' % (i+1) for i in range(numPlots)]
    else:
      assert len(axTitles) == numPlots, len(axTitles)
    self.fig        = plt.figure()
    self.axs        = []
    self.image_objs = []
    self.line_objs  = []
    self.bgs        = []
    self.numPlots   = numPlots 
    self.axTitles   = axTitles
    im = np.zeros((height, width, 3)).astype(np.uint8)
    plt.show(block=False)
    for i in range(numPlots):
      shp = subPlotShape + (i+1,)
      aa  = self.fig.add_subplot(*shp)
      self.axs.append(aa)
      if isIm[i]:
        self.image_objs.append(self.axs[i].imshow(im))
        self.line_objs.append([])
      else:
        self.line_objs.append(self.axs[i].plot(range(200), 200*[0])[0])
        self.image_objs.append([])
      if axTitles is not None:
        self.axs[i].set_title(axTitles[i])
        self.axs[i].autoscale(True)
      self.fig.canvas.draw()
      self.bgs.append(self.fig.canvas.copy_from_bbox(self.axs[i].bbox))

  def __del__(self):
    plt.close(self.fig)

  def _display(self, pixels):
    """
    Args:
      pixels: list of either images
              or objects of the type
              matplotlib.lines.Line2D etc.
    """
    assert type(pixels) in [list, tuple]
    for i, pix in enumerate(pixels):
      self.fig.canvas.restore_region(self.bgs[i])
      if type(pix) is np.ndarray:
        self.image_objs[i].set_data(pix)
        self.axs[i].draw_artist(self.image_objs[i])
        self.fig.canvas.blit(self.axs[i].bbox)
      else:
        self.axs[i].clear()
        self.axs[i].set_title(self.axTitles[i])
        self.axs[i].plot(*pix)
        #self.line_objs[i].set_data(*pix)
        #self.axs[i].draw_artist(self.line_objs[i])
    plt.pause(0.01)


##
#Make a video
class VideoMaker(object):
  def __init__(self, vidName='video.mp4'):
    self.vidName = vidName
    path_utils.mkdirs_in_path(vidName)       
    self.tmpDir = '_tmp_vid_%s_%d' % (int(time.time()), np.random.randint(50000))
    path_utils.mkdir(self.tmpDir)
    self.count  = 0

  def save_frame(self, im):
    """
    save im as the frame
    """ 
    imName = osp.join(self.tmpDir, '%d.png' % self.count)
    self.count += 1
    scm.imsave(imName, im)

  def compile_video(self, fps=30, imSz=None):
    cmd = "ffmpeg -i {0}/%d.png -b 10000k -c:v libx264".format(self.tmpDir)
    if fps:
      cmd = "{0} -r {1}".format(cmd, int(fps))
    if imSz:
      w, h = imSz
      cmd = "{0} -s {1}x{2}".format(cmd, int(w), int(h))
    os.system("{0} {1}".format(cmd, self.vidName))
    shutil.rmtree(self.tmpDir) 


def draw_square_on_im(im, sq, col='k', conf=None):
  """
  Draw a box on an image.

  Args:
  im: an np.ndarray image
  sq: box coords [x1, y1, x2, y2]
  col: box color
  conf: if not None, box confidence

  Returns:
  im: the processed image
  """
  im = Image.fromarray(im, 'RGB')

  w, h = im.size
  x1, y1, x2, y2 = sq
  x1 = max(0, int(x1))
  y1 = max(0, int(y1))
  x2 = min(int(x2), w-1)
  y2 = min(int(y2), h-1)

  if col in ['k', 'black']:
    col = (0, 0, 0)
  elif col in ['b', 'blue']:
    col = (0, 0, 255)
  elif col in ['g', 'green']:
    col = (0, 255, 0)
  elif col in ['r', 'red']:
    col = (255, 0, 0)
  elif col in ['c', 'cyan']:
    col = (0, 255, 255)
  elif col in ['m', 'magenta']:
    col = (255, 0, 255)
  elif col in ['y', 'yellow']:
    col = (255, 255, 0)
  elif col in ['w', 'white']:
    col = (255, 255, 255)
  
  draw = ImageDraw.Draw(im)
  draw.rectangle([x1, y1, x2, y2], outline=col)

  if conf is not None:
    conf = '{:.3f}'.format(conf)
    fontFile = osp.join(osp.dirname(__file__), 'tom-thumb.pil')
    font = ImageFont.load(fontFile)
    textW, textH = draw.textsize(conf, font=font)
    draw.rectangle([x1, y1, x1+textW, y1+textH], outline=col, fill=col)
    if col in [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]:
        textCol = (255, 255, 255)
    else:
        textCol = (0, 0, 0)
    draw.text([x1+1, y1+1], conf, fill=textCol, font=font)

  return np.array(im)


def visualize_grid(data, ubound=255.0, padding=1):
  """
  Arrange a 4D tensor of image data in a grid for easy visualization.

  Args:
  data: Data of shape (n, c, h, w)
  ubound: Output grid will have values scaled to the range [0, ubound]
  padding: The number of blank pixels between elements of the grid

  Returns:
  grid: an image with each of the n (h, w, c) images arranged in a grid.
  """
  (n, c, h, w) = data.shape
  grid_size = int(ceil(sqrt(n)))
  grid_height = h * grid_size + padding * (grid_size - 1)
  grid_width = w * grid_size + padding * (grid_size - 1)
  grid = np.zeros((grid_height, grid_width, c))
  next_idx = 0
  y0, y1 = 0, h
  for y in xrange(grid_size):
    x0, x1 = 0, w
    for x in xrange(grid_size):
      if next_idx < n:
        img = data[next_idx, ...].transpose(1, 2, 0)
        low, high = np.min(img), np.max(img)
        grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
        next_idx += 1
      x0 += w + padding
      x1 += w + padding
    y0 += w + padding
    y1 += w + padding
  return grid
