import os
from os import path as osp

##
#
def mkdir(dirName):
  if len(dirName) > 0:
  	if not osp.exists(dirName):
	  	os.makedirs(dirName)


def mkdirs_in_path(fName):
  """
    make directories in path of fName
  """
  mkdir(osp.dirname(fName))
  
