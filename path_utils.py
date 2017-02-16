import os
from os import path as osp

##
#
def mkdir(fName):
	if not osp.exists(fName):
		os.makedirs(fName)


