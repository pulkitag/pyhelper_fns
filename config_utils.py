import abc
import copy
import os
import yaml

class ConfigHelper(object):
  """
  Helper class for managing configuration files. 

  :param opts: configuration options.
  :var allOptKeys: list of all keys to configure options.
  """

  __metaclass__ = abc.ABCMeta

  def __init__(self, opts={}):
    assert type(opts) is dict, "Options should be provided as dict."
    self._opts = copy.deepcopy(opts)

  @abc.abstractproperty
  def allOptKeys(self):
    """Returns the set of all keys that can be configured."""

  @property
  def opts(self):
    return self._opts

  def optKeys(self):
    return self.opts.keys()

  def update_from_yaml(self, fName):
    if os.path.exists(fName):
      with open(fName, 'rb') as f:
        dat = yaml.load(f)
        for k in dat.keys():
          dat[k] = str(dat[k])
        self._opts.update(dat)
      

  def update_from_shell(self):
    """Update the options from shell variables """
    for k in self.allOptKeys:
      if os.getenv(k):
        self._opts.update({k: os.getenv(k)})

  def verify(self):
    """ Verifies that all keys have been populated. """
    isTrue = True
    for k in self.allOptKeys:
      isTrue = isTrue and k in self.opts
    return isTrue
