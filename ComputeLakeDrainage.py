#!/usr/bin/env python

from __future__ import absolute_import, division, print_function
from netCDF4 import Dataset


def LakeDrainage(fIn, tind = 0, rho_i = 910., rho_w = 1000., N_neighbors = 4):
  import ctypes
  import LakeDrainage as LD

  print ("Reading file "+fIn+" ...")
  ncIn = Dataset(fIn, 'r')

  topg = getNcVarSlice(ncIn, 'topg', tind)
  shape = topg.shape

  try:
    x = (ncIn.variables['x'][:]).astype("double")
  except:
    x = np.arange(0, shape[1])

  dx = (x[1] - x[0]).astype("double")
  cell_area = dx * dx

  try:
    y = (ncIn.variables['y'][:]).astype("double")
  except:
    y = np.arange(0, shape[0])

  try:
    topg_filtered = getNcVarSlice(ncIn, 'topg_filtered', tind, shape)
  except:
    print("   -> Setting it to topg")
    topg_filtered = topg

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

  result = LD.LakeDrainage(x, y, depth, topg_filtered, thk, ocean_mask, cell_area, rho_i, rho_w, int(N_neighbors))

  return result



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

  result = LakeDrainage(options.fIn, options.tind, options.rhoi, options.rhow)

  shape = [len(result.y), len(result.x)]

  area = np.zeros(shape)
  volume = np.zeros(shape)
  max_depth = np.zeros(shape)

  for i in range(len(result.area)):
    area_i = result.area[i]
    volume_i = result.volume[i]
    max_depth_i = result.max_depth[i]

    i_mask = (result.lake_mask == i)

    area[i_mask] = area_i/(1000. * 1000.)
    volume[i_mask] = volume_i/(1000. * 1000. * 1000.)
    max_depth[i_mask] = max_depth_i

  ncOut = Dataset('out.nc', 'w')

  xDim = ncOut.createDimension('x', len(result.x))
  yDim = ncOut.createDimension('y', len(result.y))

  x_out = ncOut.createVariable('x','f4', ['x'])
  x_out.units = "m"
  y_out = ncOut.createVariable('y','f4', ['y'])
  y_out.units = "m"

  x_out[:] = result.x[:]
  y_out[:] = result.y[:]

  depth_out = ncOut.createVariable('depth','f4', ['y','x'])
  depth_out[:] = result.depth[:,:]
  depth_out.units = "m"

  lake_ids_out = ncOut.createVariable('lake_ids','i', ['y','x'], fill_value=-1)
  lake_ids_out[:] = result.lake_mask[:,:]
  lake_ids_out.units = "1"

  area_out = ncOut.createVariable('area','f4', ['y','x'])
  area_out[:] = area[:,:]
  area_out.units = "km2"

  vol_out = ncOut.createVariable('volume','f4', ['y','x'])
  vol_out[:] = volume[:,:]
  vol_out.units = "km3"

  max_depth_out = ncOut.createVariable('max_depth','f4', ['y','x'])
  max_depth_out[:] = max_depth[:,:]
  max_depth_out.units = "m"

  basin_id_out = ncOut.createVariable('basin_id','i', ['y','x'])
  basin_id_out[:] = result.basin_id[:,:]
  basin_id_out.units = "1"

  drain_dir_out = ncOut.createVariable('drain_dir','i', ['y','x'])
  drain_dir_out[:] = result.drain_dir[:,:]
  drain_dir_out.units = "1"

  ncOut.close()


def parse_args():
  from argparse import ArgumentParser
  import os

  parser = ArgumentParser()
  parser.description = "Find drainage route of lakes"
  parser.add_argument("-i", "--input",  dest="fIn",  required=True, help="Input file", metavar="FILE", type=lambda x: is_valid_file(parser, x))
  parser.add_argument('-tind', "--time-index", dest="tind", help="index of time dimension", default=-1, type=int)
  parser.add_argument('-rho_i', "--ice_density", dest="rhoi", help="Density of ice", default=910., type=float)
  parser.add_argument('-rho_w', "--fresh_water_density", dest="rhow", help="Density of fresh water", default=1000., type=float)

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

