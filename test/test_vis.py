from pyhelper_fns.vis_utils import MyAnimation
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
  for i in range(10):
    canvas._display(img[i % 3])
    plt.show()
