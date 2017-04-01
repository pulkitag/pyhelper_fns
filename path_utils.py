import os
from os import path as osp

##
#
def mkdir(dirName):
	if not osp.exists(dirName):
		os.makedirs(dirName)


def mkdirs_in_path(fName):
  """
    make directories in path of fName
  """
  dirName = mkdir(osp.dirname(fName))  
