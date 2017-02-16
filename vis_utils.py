## @package vis_utils
#  Miscellaneous Functions for visualizations
#
import numpy as np
import scipy.misc as scm
import matplotlib.pyplot as plt
import copy
import os
import pdb
from matplotlib import gridspec

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
      
  def _display(self, pixels):
    self.image_obj.set_data(pixels)
    self.fig.canvas.restore_region(self.bg)
    self.ax.draw_artist(self.image_obj)
    self.fig.canvas.blit(self.ax.bbox)


def draw_square_on_im(im, sq, width=4, col='w'):
  x1, y1, x2, y2 = sq
  h = im.shape[0]
  w = im.shape[1]
  if col == 'w':
    col = (255 * np.ones((1,1,3))).astype(np.uint8)
  elif col == 'r':
    col = np.zeros((1,1,3))
    col[0,0,0] = 255
    col =  col.astype(np.uint8)
  #Top Line
  im[max(0,int(y1-width/2)):min(h, y1+int(width/2)),x1:x2,:] = col
  #Bottom line
  im[max(0,int(y2-width/2)):min(h, y2+int(width/2)),x1:x2,:] = col
  #Left line
  im[y1:y2, max(0,int(x1-width/2)):min(h, x1+int(width/2))] = col
  #Right line
  im[y1:y2, max(0,int(x2-width/2)):min(h, x2+int(width/2))] = col
  return im

