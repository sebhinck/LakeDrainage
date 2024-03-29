from netCDF4 import Dataset
import os
import pickle

import ComputeLakeDrainage

#Input file & Basin file
fIn = '/scratch/users/shinck/IceModelling/Evan_19/lakes/merged_filtered10km_new.nc'
obIn = '/scratch/users/shinck/IceModelling/Python/LakeDrainage/Basins.nc'

#Path to write the output
path_out='/scratch/users/shinck/IceModelling/Evan_19/lakes/analysis_new'

if not os.path.isdir(path_out):
  os.mkdir(path_out)

#Name prefix of output files
Name_prefix = 'Evan19_filtered10km'

#number of time slices in input file
Nt = 31

tinds = range(Nt)

areas = [None] * Nt
volumes = [None] * Nt
lake_levels = [None] * Nt
max_depths = [None] * Nt
spillway_idx = [None] * Nt
drain_basin_id = [None] * Nt



ncOutName = os.path.join(path_out, Name_prefix+"_drainage.nc")

ncOut = Dataset(ncOutName, "w")

with Dataset(fIn) as src:
    # copy global attributes all at once via dictionary
    ncOut.setncatts(src.__dict__)
    # copy dimensions
    for name, dimension in src.dimensions.items():
        ncOut.createDimension(
            name, (len(dimension) if not dimension.isunlimited() else None))
    # copy all file data except for the excluded
    for name, variable in src.variables.items():
        if name in ["x", "y", "t"]:
            x = ncOut.createVariable(name, variable.datatype, variable.dimensions)
            ncOut[name][:] = src[name][:]
            # copy variable attributes all at once via dictionary
            ncOut[name].setncatts(src[name].__dict__)



print(ncOut.dimensions)

topg_out = ncOut.createVariable('topg','f4', ['t','y','x'])
topg_out.units = "m"

thk_out = ncOut.createVariable('thk','f4', ['t','y','x'])
thk_out.units = "m"

usurf_filtered_out = ncOut.createVariable('usurf_filtered','f4', ['t','y','x'])
usurf_filtered_out.units = "m"

surf_eff_out = ncOut.createVariable('surf_eff','f4', ['t','y','x'])
surf_eff_out.units = "m"

ocean_mask_out = ncOut.createVariable('ocean_mask','i', ['t','y','x'])
ocean_mask_out.units = "1"

depth_out = ncOut.createVariable('depth','f4', ['t','y','x'])
depth_out.units = "m"

lake_ids_out = ncOut.createVariable('lake_ids','i', ['t','y','x'], fill_value=-1)
lake_ids_out.units = "1"

basin_id_out = ncOut.createVariable('basin_id','i', ['t','y','x'])
basin_id_out.units = "1"

drain_dir_out = ncOut.createVariable('drain_dir','i', ['t','y','x'])
drain_dir_out.units = "1"



for tind in tinds:
    print(tind)
    
    result = ComputeLakeDrainage.LakeDrainage(fIn, obIn, tind=tind, N_neighbors=4)
    
    topg_out[tind,:,:] = result.topg[:]
    thk_out[tind,:,:] = result.thk[:]
    depth_out[tind,:,:] = result.depth[:]
    usurf_filtered_out[tind,:,:] = result.usurf_filtered[:]
    surf_eff_out[tind,:,:] = result.surf_eff[:]
    ocean_mask_out[tind,:,:] = result.ocean_mask[:]
    lake_ids_out[tind,:,:] = result.lake_mask[:]
    basin_id_out[tind,:,:] = result.basin_id[:]
    drain_dir_out[tind,:,:] = result.drain_dir[:]

    areas[tind] = result.area.copy()
    volumes[tind] = result.volume.copy()
    lake_levels[tind] = result.lake_level.copy()
    max_depths[tind] = result.max_depth.copy()
    spillway_idx[tind] = result.spillway_idx.copy()
    drain_basin_id[tind] = result.drain_basin_id.copy()

ncOut.close()

pickleLakeName  = os.path.join(path_out, Name_prefix+"_lakes.pickle")
pickleBasinName = os.path.join(path_out, Name_prefix+"_basins.pickle")

data_lakes = {'areas': areas, 'volumes': volumes, 'lake_levels': lake_levels, 'max_depths': max_depths}
with open(pickleLakeName, 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(data_lakes, f, pickle.HIGHEST_PROTOCOL)

data_basins = {'spillway_idx': spillway_idx, 'drain_basin_id': drain_basin_id}
with open(pickleBasinName, 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(data_basins, f, pickle.HIGHEST_PROTOCOL)
