## @package vis_utils
#  Miscellaneous Functions for visualizations
#
import numpy as np
import scipy.misc as scm
import matplotlib.pyplot as plt
import copy
import os
from os import path as osp
import pdb, time, shutil
from matplotlib import gridspec
from functools import reduce
from pyhelper_fns import path_utils

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
        #self.axs[i].autoscale(True)
        self.line_objs.append(self.axs[i].plot(range(200), 200*[0])[0])
        #self.axs[i].autoscale(True)
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


def draw_square_on_im(im, sq, width=4, col='k'):
  x1, y1, x2, y2 = sq
  x1 = max(0, int(x1))
  y1 = max(0, int(y1))
  x2 = int(np.floor(x2))
  y2 = int(np.floor(y2))
  h = im.shape[0]
  w = im.shape[1]
  if col in ['k', 'black']:
    col = np.zeros((1,1,3), dtype=np.uint8)
  elif col in ['r', 'red']:
    col = np.zeros((1,1,3), dtype=np.uint8)
    col[0,0,0] = 255
  elif col in ['g', 'green']:
    col = np.zeros((1,1,3), dtype=np.uint8)
    col[0,0,1] = 255
  elif col in ['b', 'blue']:
    col = np.zeros((1,1,3), dtype=np.uint8)
    col[0,0,2] = 255
  elif col in ['y', 'yellow']:
    col = np.zeros((1,1,3), dtype=np.uint8)
    col[0,0,0] = 255
    col[0,0,1] = 255
  elif col in ['m', 'magenta']:
    col = np.zeros((1,1,3), dtype=np.uint8)
    col[0,0,0] = 255
    col[0,0,2] = 255
  elif col in ['c', 'cyan']:
    col = np.zeros((1,1,3), dtype=np.uint8)
    col[0,0,1] = 255
    col[0,0,2] = 255
  elif col in ['w', 'white']:
    col = np.zeros((1,1,3), dtype=np.uint8)
    col[0,0,0] = 255
    col[0,0,1] = 255
    col[0,0,2] = 255
  #Top Line
  im[max(0,int(y1-width/2)):min(h, y1+int(width/2)),x1:x2,:] = col
  #Bottom line
  im[max(0,int(y2-width/2)):min(h, y2+int(width/2)),x1:x2,:] = col
  #Left line
  im[y1:y2, max(0,int(x1-width/2)):min(w, x1+int(width/2))] = col
  #Right line
  im[y1:y2, max(0,int(x2-width/2)):min(w, x2+int(width/2))] = col
  return im
