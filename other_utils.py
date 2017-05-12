import numpy as np
import copy

def update_default_params(defaults, params={}):
  """
  Args:
    defaults: dict of default params
    params  : the params to update the defaults
  """
  assert type(defaults) is dict
  assert type(params) is dict
  assert params.keys().issubset(defaults.keys())
  newParams = copy.deepcopy(defaults)
  newParams.update(params)
  return newParams
