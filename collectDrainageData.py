import pandas as pd
from netCDF4 import Dataset
import os
import numpy as np
import pickle
import shapefile as shpf


###############################################################################
### Define variables ##########################################################
###############################################################################

path_out='/scratch/users/shinck/IceModelling/Evan_19/lakes/analysis_new'
Name_prefix = 'Evan19_filtered10km'

Nt=31

t_name=list()
t=list()
for i in range(Nt):
    year = Nt-i - 1
    t.append(year)
    t_name.append(str(year)+"ka")

data = pd.DataFrame(index=t_name)

#data['Agassiz'] = pd.Series({'0ka': (9,3), '20ka': (2,3)})
data['HuronMichigan'] = pd.Series({'0ka': (1007,193),
                                   '1ka': (1007,193),
                                   '2ka': (1007,193),
                                   '3ka': (1007,193),
                                   '4ka': (1007,193),
                                   '5ka': (1007,193),
                                   '6ka': (1007,193),
                                   '7ka': (1007,193),
                                   '8ka': (1007,193),
                                   '9ka': (1007,193),
                                   '13ka': (1007,193),
                                   '14ka': (1007,193)
})

data['Huron'] = pd.Series({'10ka': (1007,193),
                           '11ka': (1007,193)
})

data['Michigan'] = pd.Series({'10ka': (1007,193),
                              '11ka': (1007,193),
                              '15ka': (935,150)
})

data['SuperiorHuronMichigan'] = pd.Series({'12ka': (1000,214)
})
                                   
data['Superior'] = pd.Series({'0ka': (930,252),
                              '1ka': (930,252),
                              '2ka': (930,252),
                              '3ka': (930,252),
                              '4ka': (930,252),
                              '5ka': (930,252),
                              '6ka': (930,252),
                              '7ka': (930,252),
                              '8ka': (930,252),
                              '9ka': (930,252),
                              '10ka': (930,252),
                              '13ka': (862,236)
})

data['Erie'] = pd.Series({'0ka': (1035,150),
                          '1ka': (1035,150),
                          '2ka': (1035,150),
                          '3ka': (1035,150),
                          '4ka': (1035,150),
                          '5ka': (1035,150),
                          '6ka': (1035,150),
                          '7ka': (1035,150),
                          '8ka': (1035,150),
                          '9ka': (1035,150),
                          '10ka': (1035,150),
                          '11ka': (1035,150),
                          '12ka': (1057,164),
                          '13ka': (1057,164),
                          '14ka': (1060,164),
                          '17ka': (1013,136)
})

data['HuronErie'] = pd.Series({'15ka': (1013,152),
                               '16ka': (1022,150)
})

data['Ontario'] = pd.Series({'0ka': (1088,192),
                             '1ka': (1088,192),
                             '2ka': (1088,192),
                             '3ka': (1088,192),
                             '4ka': (1088,192),
                             '5ka': (1088,192),
                             '6ka': (1088,192),
                             '7ka': (1088,192),
                             '8ka': (1088,192),
                             '9ka': (1088,192),
                             '10ka': (1088,192),
                             '11ka': (1088,192),
                             '12ka': (1088,192),
                             '13ka': (1088,192),
                             '14ka': (1088,192)
})

data['Nipigon'] = pd.Series({'1ka': (900,304),
                             '2ka': (900,304),
                             '3ka': (900,304),
                             '4ka': (900,304),
                             '5ka': (900,304),
                             '6ka': (900,304),
                             '7ka': (900,304),
                             '8ka': (900,304),
                             '9ka': (900,304),
                             '10ka': (900,304)
})

data['StClair'] = pd.Series({'0ka': (1011,150),
                             '1ka': (1011,150),
                             '2ka': (1011,150),
                             '3ka': (1011,150),
                             '4ka': (1011,150),
                             '5ka': (1011,150),
                             '6ka': (1011,150),
                             '7ka': (1011,150),
                             '8ka': (1011,150),
                             '9ka': (1011,150),
                             '10ka': (1011,150),
                             '11ka': (1009,151)
})

data['Bonneville'] = pd.Series({'0ka': (489,144),
                                '1ka': (489,144),
                                '2ka': (489,144),
                                '3ka': (489,144),
                                '4ka': (489,144),
                                '5ka': (489,144),
                                '6ka': (489,144),
                                '7ka': (489,144),
                                '8ka': (489,144),
                                '9ka': (489,144),
                                '10ka': (489,144),
                                '11ka': (489,144),
                                '12ka': (489,144),
                                '13ka': (489,144),
                                '14ka': (489,144),
                                '15ka': (489,144),
                                '16ka': (489,144),
                                '17ka': (489,144),
                                '18ka': (489,144),
                                '19ka': (489,144),
                                '20ka': (489,144),
                                '21ka': (489,144)
})


data['Lahontan'] = pd.Series({'0ka': (400,150),
                              '1ka': (400,150),
                              '2ka': (400,150),
                              '3ka': (400,150),
                              '4ka': (400,150),
                              '5ka': (400,150),
                              '6ka': (400,150),
                              '7ka': (400,150),
                              '8ka': (400,150),
                              '9ka': (400,150),
                              '10ka': (400,150),
                              '11ka': (400,150),
                              '12ka': (400,150),
                              '13ka': (400,150),
                              '14ka': (400,150),
                              '15ka': (400,150),
                              '16ka': (400,150),
                              '17ka': (400,150),
                              '18ka': (400,150),
                              '19ka': (400,150),
                              '20ka': (400,150),
                              '21ka': (400,150)
})



data['Athabasca'] = pd.Series({'0ka': (641,530),
                               '1ka': (641,530),
                               '2ka': (641,530),
                               '3ka': (641,530),
                               '4ka': (641,530),
                               '5ka': (641,530),
                               '6ka': (641,530),
                               '7ka': (641,530),
                               '8ka': (641,530),
                               '9ka': (641,530),
                               '10ka': (641,530)
})


data['GreatBear'] = pd.Series({'0ka': (588,706),
                               '1ka': (588,706),
                               '2ka': (588,706),
                               '3ka': (588,706),
                               '4ka': (588,706),
                               '5ka': (588,706),
                               '6ka': (588,706),
                               '7ka': (588,706),
                               '8ka': (588,706),
                               '9ka': (588,706),
                               '10ka': (604,702),
                               '11ka': (604,702),
                               '13ka': (592,706),
                               '14ka': (592,706)
})


data['GreatSlave'] = pd.Series({'1ka': (600,590),
                                '2ka': (600,590),
                                '3ka': (600,590),
                                '4ka': (600,590),
                                '5ka': (600,590),
                                '6ka': (600,590),
                                '7ka': (600,590),
                                '8ka': (600,590),
                                '9ka': (600,590)
})

data['Agassiz'] = pd.Series({'9ka': (770,384),
                             '10ka': (770,338),
                             '11ka': (770,338),
                             '12ka': (770,308),
                             '13ka': (781,284),
                             '14ka': (781,280),
                             '15ka': (781,256)
})

data['McConnell'] = pd.Series({'10ka': (610,595),
                               '11ka': (610,595),
                               '12ka': (600,600)
})

data['McConnell_outside'] = pd.Series({'13ka': (561,531),
                                       '14ka': (505,577),
                                       '15ka': (505,577),
                                       '16ka': (507,584),
                                       '21ka': (525,600)
})

data['McConnell_outside2'] = pd.Series({'14ka': (588,445),
                                        '15ka': (600,406)
})

data['McKenzie'] = pd.Series({'14ka': (536,691)
})

data['Ojibway'] = pd.Series({'9ka': (945,322)
})

data['Ojibway2'] = pd.Series({'9ka': (1021,365)
})

data['Reindeer'] = pd.Series({'1ka': (729,480),
                              '2ka': (729,480),
                              '3ka': (729,480),
                              '4ka': (729,480),
                              '5ka': (729,480),
                              '6ka': (729,480),
                              '7ka': (729,480),
                              '8ka': (729,480)
})

print(data)

epsg='PROJCS["Lambert_Azimuthal_Equal_Area",\
             GEOGCS["GCS_WGS_1984",DATUM["D_unknown",SPHEROID["WGS84",6378137,298.257223563]],\
             PRIMEM["Greenwich",0],\
             UNIT["Degree",0.017453292519943295]],\
             PROJECTION["Lambert_Azimuthal_Equal_Area"],\
             PARAMETER["latitude_of_origin",60],\
             PARAMETER["central_meridian",-94],\
             PARAMETER["false_easting",4104009.407173712],\
             PARAMETER["false_northing",2625682.633840935],\
             UNIT["Meter",1]]'

###############################################################################
### Read data #################################################################
###############################################################################
ncInName = os.path.join(path_out, Name_prefix+"_drainage.nc")
pickleLakeName  = os.path.join(path_out, Name_prefix+"_lakes.pickle")
pickleBasinName = os.path.join(path_out, Name_prefix+"_basins.pickle")

ncIn = Dataset(ncInName, 'r')

try:
    T = ncIn.variables['t'][:]
except:
    print("t not found in file.")
    raise

try:
    x = ncIn.variables['x'][:]
except:
    print("x not found in file.")
    raise

try:
    y = ncIn.variables['y'][:]
except:
    print("y not found in file.")
    raise

try:
    lake_ids = ncIn.variables['lake_ids'][:,:,:]
except:
    print("lake_ids not found in file.")
    raise    

ncIn.close()

xDim = len(x)
yDim = len(y)

with open(pickleLakeName, 'rb') as f:
    lakeData = pickle.load(f)

with open(pickleBasinName, 'rb') as f:
    basinData = pickle.load(f)

###############################################################################
### Define functions ##########################################################
###############################################################################

def ind2idx(_x, _y):
    _idx = _y * xDim + _x

    if (_idx < 0) or (_idx >= (xDim * yDim)):
        raise ValueError("Indices ("+str(_x)+", "+str(_y)+") out of bounds!")

    return _idx;

def idx2ind(_idx):
    if (_idx < 0) or (_idx >= (xDim * yDim)):
        raise ValueError("Index "+str(_idx)+" out of bounds!")

    _y = _idx // xDim
    _x = _idx - (xDim * _y)

    return (_x, _y)

def getDrainageRoute(_basin_id, Nmax = 10000):
    path = list()
    
    N = 0
    
    basin_id = _basin_id
    
    while ((basin_id >= 0) and (N < Nmax)):
        #print(basin_id)
        spillway_idx = basinData['spillway_idx'][t_idx][basin_id]
        spillway_ind = idx2ind(spillway_idx)
        path.append([x[spillway_ind[0]], y[spillway_ind[1]]])
        
        basin_id = basinData['drain_basin_id'][t_idx][basin_id]
        N=N+1
    
    return (basin_id, [path])
    

###############################################################################
### Loop ######################################################################
###############################################################################
shpName = os.path.join(path_out, Name_prefix)
tableName = os.path.join(path_out, Name_prefix+"_summary.txt")

#Write Projection file
with open("%s.prj" % shpName, "w") as prj:
    prj.write(epsg)


with shpf.Writer(shpName, shapeType=shpf.POLYLINE) as shp, open(tableName, "w") as tab:
    shp.field('name', 'C')
    shp.field('kaBP', 'N')

    tab.write("Name\tYear[kaBP]\tArea[km^2]\tVolume[km^3]\tLevel[m]\tMax depth[m]\tSink\n")

    for lake in data.keys():
        for t_idx in range(Nt):
            if not np.isnan(data[lake][t_idx]).any():
                print(lake+": "+t_name[t_idx])
                
                ind = data[lake][t_idx]
                
                idx = ind2idx(ind[0], ind[1])
                
                lake_id = lake_ids.data[t_idx, ind[1], ind[0]]
                
                if (lake_id == -1):
                    msg="Error, Lake "+lake+" at t="+t_name[t_idx]+" not existing!"
                    print(msg)
                    raise(ValueError)
                
                coord = (x[ind[0]], y[ind[1]])
                
                lake_spillway_idx = basinData['spillway_idx'][t_idx][lake_id]
                lake_spillway_ind = idx2ind(lake_spillway_idx)
                lake_area = lakeData['areas'][t_idx][lake_id]/(1000**2)
                lake_vol  = lakeData['volumes'][t_idx][lake_id]/(1000**3)
                lake_level     = lakeData['lake_levels'][t_idx][lake_id]
                lake_max_depth = lakeData['max_depths'][t_idx][lake_id]

                sink, route =getDrainageRoute(lake_id)
                
                sink_name='NONE'
                if sink == -16:
                    sink_name = 'ATLANTIC'
                elif sink == -15:
                    sink_name = 'STLAWRENCE'
                elif sink == -14:
                    sink_name = 'HUDSONBAY'
                elif sink == -13:
                    sink_name = 'CANARCHIPEL'
                elif sink == -12:
                    sink_name = 'ARCTIC'
                elif sink == -11:
                    sink_name = 'BERINGS'
                elif sink == -10:
                    sink_name = 'PACIFIC'
                elif sink == -7:
                    sink_name = 'OCEAN'
                elif sink == -6:
                    sink_name = 'NORTH'
                elif sink == -5:
                    sink_name = 'EAST'
                elif sink == -4:
                    sink_name = 'SOUTH'
                elif sink == -3:
                    sink_name = 'WEST'
                elif sink == -2:
                    sink_name = 'LOOP'
                else:
                    sink_name = 'UNDEFINED'

                shp.record(name=lake, kaBP=t[t_idx])
                shp.line(route)

                tab.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(lake, t[t_idx], lake_area, lake_vol, lake_level, lake_max_depth, sink_name))
