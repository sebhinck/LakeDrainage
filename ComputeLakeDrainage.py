#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

import LakeDrainage as LD
import ctypes


def LakeDrainage(fIn, tind):
  from netCDF4 import Dataset
  #import LakeCC as LCC

  print ("Reading file "+fIn+" ...")
  ncIn = Dataset(fIn, 'r')

  topg = getNcVarSlice(ncIn, 'topg', tind)
  shape = topg.shape

  try:
    x = ncIn.variables['x'][:]
  except:
    x = np.arange(0, shape[1])

  dx = (x[1] - x[0]).astype("double")
  cell_area = dx * dx

  try:
    y = ncIn.variables['y'][:]
  except:
    y = np.arange(0, shape[0])

  try:
    thk = getNcVarSlice(ncIn, 'thk', tind, shape)
  except:
    print("   -> Setting it to zero")
    thk = np.zeros(shape)
    
  try:
    depth = getNcVarSlice(ncIn, 'lake_depth', tind, shape)
  except:
    print("   -> Setting it to zero")
    depth = np.zeros(shape)

  try:
    ocean_mask = getNcVarSlice(ncIn, 'ocean_mask', tind, shape)
  except:
    ocean_mask = np.zeros(shape)
    ocean_mask[topg < 0] = 1

  ocean_mask = ocean_mask.astype(ctypes.c_int)

  ncIn.close()

  test = LD.LakeDrainage(depth, topg, thk, ocean_mask, cell_area)

  print(test.lake_mask)
  print(test.area)
  print(test.volume)


def getNcVarSlice(nc, varname, tind = -1, shape = None):
  try:
    var = nc.variables[varname]
    dims = var.dimensions
    if len(dims) == 2:
      data = var[:,:]
    elif len(dims) == 3:
      data = var[tind, :, :]
    else:
      raise ValueError("Wrong number of dimensions: "+str(len(dims)))
  except:
    print("-" + varname + " not found in file.")
    raise

  if shape is not None:
    if shape != data.shape:
      raise ValueError("Dimensions of "+varname+ "do not match required dimensions.")

  return data.astype("double")

def main():
  options = parse_args()

  LakeDrainage(options.fIn, options.tind)

def parse_args():
  from argparse import ArgumentParser
  import os

  parser = ArgumentParser()
  parser.description = "Find drainage route of lakes"
  parser.add_argument("-i", "--input",  dest="fIn",  required=True, help="Input file", metavar="FILE", type=lambda x: is_valid_file(parser, x))
  parser.add_argument('-tind', "--time-index", dest="tind", help="index of time dimension", default=-1, type=int)

  options = parser.parse_args()
  return options

def is_valid_file(parser, arg):
  import os

  if not os.path.exists(arg):
    parser.error("The file %s does not exist!" % arg)
  else:
    return os.path.abspath(arg)  # return file path



if __name__ == "__main__":
  import numpy as np
  main()

