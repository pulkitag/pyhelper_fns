import os
from os import path as osp

#Check is a path exist, else raise exception
def exists(pathname, silent=False):
  """
  Args:
    pathname: path that needs to be checked
    silent  : True raise an exception if path doesnot exist
            : False print a warning if path doesnot exist 
  """
  if not osp.exists(pathname):
    if silent:
      print ('Path %s doesnot exist' % pathname)
    else:
      raise Exception('Path %s doesnot exist' % pathname) 


  
