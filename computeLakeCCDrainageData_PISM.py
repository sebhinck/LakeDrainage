from netCDF4 import Dataset
import os
import pickle
import numpy as np
import ComputeLakeDrainage

#Input file & Basin file
fIn = '/home/shinck/projects/PISMLakeCC_paper/data/Ind_lin2_Lake_lakecc_extra_100years.nc'
obIn = '/home/shinck/projects/Tools/LakeDrainage/Basins_PISM.nc'
fOverlay = '/home/shinck/projects/PISMLakeCC_paper/data/topg_smooth_overlay_filtered10km_bil_laurentide_20km.nc'

#Path to write the output
path_out='/home/shinck/projects/PISMLakeCC_paper/data/LakeDrainage/'

if not os.path.isdir(path_out):
  os.mkdir(path_out)

#Name prefix of output files
Name_prefix = 'Ind_lin2_Lake_lakecc'


simtime = lambda t: (21 - t) * -1e3

times_kyrBP = np.array([8.5, 11, 14, 17.5, 18, 22])
times = simtime(times_kyrBP)

Nt = len(times)

areas = [None] * Nt
volumes = [None] * Nt
lake_levels = [None] * Nt
max_depths = [None] * Nt
spillway_idx = [None] * Nt
drain_basin_id = [None] * Nt

print(times)

ncOutName = os.path.join(path_out, Name_prefix+"_drainage.nc")

ncOut = Dataset(ncOutName, "w")

with Dataset(fIn) as src:

    t = src['time'][:]

    tinds = [np.where(t == x)[0][0] for x in times]

    # copy global attributes all at once via dictionary
    ncOut.setncatts(src.__dict__)
    # copy dimensions
    for name, dimension in src.dimensions.items():
        if name == 'time':
            dim_len = Nt
        else:
            dim_len = (len(dimension) if not dimension.isunlimited() else None)
        ncOut.createDimension(name, dim_len)
    # copy all file data except for the excluded
    for name, variable in src.variables.items():
        if name in ["x", "y", "time"]:
            x = ncOut.createVariable(name, variable.datatype, variable.dimensions)
            print(x) 
            # copy variable attributes all at once via dictionary
            ncOut[name].setncatts(src[name].__dict__)
            if name == 'time':
                ncOut[name][:] = src[name][tinds]
            else:
                ncOut[name][:] = src[name][:]

with Dataset(fOverlay) as ncOverlay:
    Overlay = ncOverlay['topg_overlay'][0,:,:]

    name = 'topg_overlay'
    var  = ncOverlay.variables[name] 
    Overlay_out = ncOut.createVariable(name, var.datatype, var.dimensions[-2:])
    ncOut[name].setncatts(var.__dict__)
    ncOut[name][:,:] = var[0,:,:]

    for name, variable in ncOverlay.variables.items():
        if name in ["mapping", "lon", "lat"]:
            x = ncOut.createVariable(name, variable.datatype, variable.dimensions)
            
            # copy variable attributes all at once via dictionary
            ncOut[name].setncatts(variable.__dict__)
            ncOut[name][:] = variable[:]

topg_out = ncOut.createVariable('topg','f4', ['time','y','x'])
topg_out.units = "m"

thk_out = ncOut.createVariable('thk','f4', ['time','y','x'])
thk_out.units = "m"

usurf_filtered_out = ncOut.createVariable('usurf_filtered','f4', ['time','y','x'])
usurf_filtered_out.units = "m"

surf_eff_out = ncOut.createVariable('surf_eff','f4', ['time','y','x'])
surf_eff_out.units = "m"

ocean_mask_out = ncOut.createVariable('ocean_mask','i', ['time','y','x'])
ocean_mask_out.units = "1"

depth_out = ncOut.createVariable('depth','f4', ['time','y','x'])
depth_out.units = "m"

lake_ids_out = ncOut.createVariable('lake_ids','i', ['time','y','x'], fill_value=-1)
lake_ids_out.units = "1"

basin_id_out = ncOut.createVariable('basin_id','i', ['time','y','x'])
basin_id_out.units = "1"

drain_dir_out = ncOut.createVariable('drain_dir','i', ['time','y','x'])
drain_dir_out.units = "1"

i = 0
for tind in tinds:
    print(tind)
    
    result = ComputeLakeDrainage.LakeDrainagePISM(fIn, obIn, Overlay, tind=tind, N_neighbors=4)
    
    topg_out[i,:,:] = result.topg[:]
    thk_out[i,:,:] = result.thk[:]
    depth_out[i,:,:] = result.depth[:]
    usurf_filtered_out[i,:,:] = result.usurf_filtered[:]
    surf_eff_out[i,:,:] = result.surf_eff[:]
    ocean_mask_out[i,:,:] = result.ocean_mask[:]
    lake_ids_out[i,:,:] = result.lake_mask[:]
    basin_id_out[i,:,:] = result.basin_id[:]
    drain_dir_out[i,:,:] = result.drain_dir[:]

    areas[i] = result.area.copy()
    volumes[i] = result.volume.copy()
    lake_levels[i] = result.lake_level.copy()
    max_depths[i] = result.max_depth.copy()
    spillway_idx[i] = result.spillway_idx.copy()
    drain_basin_id[i] = result.drain_basin_id.copy()

    i += 1

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
