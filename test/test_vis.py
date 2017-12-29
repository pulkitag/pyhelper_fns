from pyhelper_fns.vis_utils import MyAnimation, MyAnimationMulti
import numpy as np
import matplotlib.pyplot as plt

def test_animation():
  r = np.zeros((32, 32, 3))
  r[:,:,0] = np.ones((32,32))
  g = np.zeros((32, 32, 3))
  g[:,:,1] = np.ones((32,32))
  b = np.zeros((32, 32, 3))
  b[:,:,2] = np.ones((32,32))

  plt.ion()
  img = [r, g, b]
  canvas = MyAnimation(None, height=32, width=32)
  for i in range(9):
    canvas._display(img[i % 3])
    plt.show()

def test_animation2():
  r = np.zeros((32, 32, 3))
  r[:,:,0] = np.ones((32,32))
  g = np.zeros((32, 32, 3))
  g[:,:,1] = np.ones((32,32))
  b = np.zeros((32, 32, 3))
  b[:,:,2] = np.ones((32,32))
  fig = plt.figure()
  ax1 = fig.add_subplot(2,1,1)
  ax2 = fig.add_subplot(2,1,2)
  rew, = ax1.plot(range(100), 0.01 * np.random.rand(100)) 
  rew2, = ax2.plot(range(200), 0.01 * np.random.rand(200))
 
  #plt.ion()
  img = [r, g, b]
  canvas = MyAnimationMulti(None, height=32, width=32, numPlots=3,
              isIm=[True, False, False], axTitles=['image', 'plot', 'plot'])
  for i in range(10):
    x = range(300)
    y = 0.01 * np.random.rand(300)
    canvas._display([img[i % 3], (x,y), (x,y)])
    plt.pause(0.2)
    #plt.show()

