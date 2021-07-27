import pandas as pd
from netCDF4 import Dataset
import os
import numpy as np
import pickle
import shapefile as shpf


###############################################################################
### Define variables ##########################################################
###############################################################################

#Define path where drainage data is stored and output should be written
path_data='/home/shinck/projects/PISMLakeCC_paper/data/LakeDrainage/'
Name_prefix = 'Ind_lin2_Lake_lakecc'


simtime = lambda t: (t/(1.e3) + 21)


###############################################################################
### Read data #################################################################
###############################################################################
ncInName = os.path.join(path_data, Name_prefix+"_drainage.nc")
pickleLakeName  = os.path.join(path_data, Name_prefix+"_lakes.pickle")
pickleBasinName = os.path.join(path_data, Name_prefix+"_basins.pickle")

with Dataset(ncInName, 'r') as ncIn:
    try:
        T = ncIn.variables['time'][:]
    except:
        print("time not found in file.")
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


xDim = len(x)
yDim = len(y)

Nt=len(T)

t_name=list()
t=list()
for ti in T:
    year = simtime(ti)
    t.append(year)
    t_name.append(str(year)+"ka")

data = pd.DataFrame(index=t_name)

#Define Lakes that were identified and provide one location of the basin

data['Agassiz'] = pd.Series({ '8.5ka': (252, 63),
                             '11.0ka': (200,100),
                             '14.0ka': (200,125),
                             '17.5ka': (250,150),
})

print(data)

epsg="""
PROJCS['Lambert Conformal Conic',
GEOGCS['NAD83',
DATUM['North_American_Datum_1983',
SPHEROID['GRS 1980',6378137,298.257222101,
AUTHORITY['EPSG','7019']],
TOWGS84[0,0,0,0,0,0,0],
AUTHORITY['EPSG','6269']],
PRIMEM['Greenwich',0,
AUTHORITY['EPSG','8901']],
UNIT['degree',0.0174532925199433,
AUTHORITY['EPSG','9122']],
AUTHORITY['EPSG','4269']],
PROJECTION['Lambert_Conformal_Conic_2SP'],
PARAMETER['standard_parallel_1',45],
PARAMETER['standard_parallel_2',65],
PARAMETER['latitude_of_origin',63],
PARAMETER['central_meridian',-100],
PARAMETER['false_easting',0],
PARAMETER['false_northing',0],
UNIT['Meter',1]]
"""



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
        # print(basin_id)
        spillway_idx = basinData['spillway_idx'][t_idx][basin_id]
        spillway_ind = idx2ind(spillway_idx)
        path.append([x[spillway_ind[0]], y[spillway_ind[1]]])
        
        basin_id = basinData['drain_basin_id'][t_idx][basin_id]
        N=N+1
    
    return (basin_id, [path])
    
def save_tabular(fname,
                 data,
                 caption='',
                 formatter=None,
                 column_format=None,
                 columns=None,
                 header = None,
                 supertabular=True,
                 twocolumn=True,
                 append=False,
                 na_rep='-'):

    if columns is None:
        columns = data.columns

    if header is None:
        header = [x for x in data.index.names]
        header.extend([x for x in columns])

    tex_str = df_filtered.to_latex(formatters=formatters,
                                   escape=False,
                                   column_format=column_format,
                                   columns=columns,
                                   na_rep=na_rep)

    tex_str = tex_str.replace('NaN', na_rep)
    tex_str = tex_str.replace('\\toprule', '\\tophline')
    tex_str = tex_str.replace('\\midrule', '\\middlehline')
    tex_str = tex_str.replace('\\bottomrule', '\\bottomhline')

    old_head = tex_str.splitlines()[1]+'\n'+tex_str.splitlines()[2]+'\n'+tex_str.splitlines()[3]+'\n'

    new_head = ' & '.join(header)+ "\\\\ \n"

    if supertabular:
        tex_str = tex_str.replace('\\begin{tabular}', '\\begin{supertabular}')
        tex_str = tex_str.replace('\\end{tabular}', '\\end{supertabular}')
        tex_str = tex_str.replace(old_head, "")
        supertab_head = "\
        \\tablefirsthead{%\n\
          \\tophline \n\
          "+ new_head +"\n\
        } \n\
        \\tablehead{% \n\
           \\multicolumn{" + str(len(header)) + "}{l}{\\textbf{Table \\thetable} ~\\textit{(Continued)}}\\\\ \n\
           \\middlehline \n\
           " + new_head + "\n \
           \\middlehline \n\
        } \n\
        \\tabletail{% \n\
           \\bottomhline \n\
           \\multicolumn{" + str(len(header)) + "}{r}{\\textit{To be continued}}\\\\ \n\
        }\\tablelasttail{ \n\
           % \n\
        } \n\
        \\topcaption{ \n\
          " + caption + " \n\
        } \n"
        supertab_head = supertab_head.replace('        ', '')

        tex_str = supertab_head + tex_str
    else:
        tex_str = tex_str.replace(old_head, '\\tophline \n'+new_head)
        table='table'
        if twocolumn:
            table = table+'*'
        indent="  "
        tex_str = tex_str.replace('\n', '\n'+2*indent)
        tex_str = tex_str.replace(tex_str.splitlines()[0], indent+tex_str.splitlines()[0])
        tex_str = tex_str.replace(tex_str.splitlines()[-2]+'\n'+tex_str.splitlines()[-1], indent+tex_str.splitlines()[-2].strip()+'\n')
        tex_str = '\\begin{'+table+'}[t] \n\
                  '+indent+'\caption{ \n\
                  '+2*indent+caption+'\n\
                  '+indent+'} \n\
                  ' + tex_str + '\
                  \\end{'+table+'}'
        tex_str = tex_str.replace('                  ','')

    if append and os.path.isfile(fname):
        file_flag = 'a'
    else:
        file_flag = 'w'


    with open(fname, file_flag) as outFile:
        if file_flag == 'a':
            outFile.write('\n\n\n')
        outFile.write(tex_str)

###############################################################################
### Check if output folder exist  #############################################
###############################################################################
try:
    os.makedirs(os.path.join(path_data, 'latex'))
except:
    pass


###############################################################################
### Loop ######################################################################
###############################################################################

#-> calculate drainage lake routes and save paths to shapefile

shpName = os.path.join(path_data, Name_prefix)
tableName = os.path.join(path_data, Name_prefix+"_summary.txt")
# tableGroupName = os.path.join(path_data, Name_prefix+"_LakeGroups.txt")

LakesDF = pd.DataFrame(columns=['Name', 'kaBP', 'Area', 'Volume', 'Level', 'max. Depth', 'Sink'])

#Write Projection file
with open("%s.prj" % shpName, "w") as prj:
    prj.write(epsg)


#Save Shape files
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

                sink, route = getDrainageRoute(lake_id)
                
                sink_name='None'
                if sink == -16:
                    sink_name = 'Atlantic'
                elif sink == -15:
                    sink_name = 'St. Lawrence'
                elif sink == -14:
                    sink_name = 'Hudson Bay'
                elif sink == -13:
                    sink_name = 'Arctic Arch.'
                elif sink == -12:
                    sink_name = 'Arctic'
                elif sink == -11:
                    sink_name = 'Bering Strait'
                elif sink == -10:
                    sink_name = 'Pacific'
                elif sink == -7:
                    sink_name = 'Ocean'
                elif sink == -6:
                    sink_name = 'North'
                elif sink == -5:
                    sink_name = 'East'
                elif sink == -4:
                    sink_name = 'South'
                elif sink == -3:
                    sink_name = 'West'
                elif sink == -2:
                    sink_name = 'Loop'
                else:
                    sink_name = 'UNDEFINED'

                shp.record(name=lake, kaBP=t[t_idx])
                shp.line(route)

                tab.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(lake, t[t_idx], lake_area, lake_vol, lake_level, lake_max_depth, sink_name))

                row = {'Name': lake, 'kaBP': t[t_idx], 'Area': lake_area, 'Volume': lake_vol, 'Level': lake_level, 'max. Depth': lake_max_depth, 'Sink': sink_name}
                LakesDF = LakesDF.append(row, ignore_index=True)

def clean_file_name(x, repl_list=[' ', ',','.','-'], repl_char=''):
    for char in repl_list:
        x = x.replace(char, repl_char)
    return x

#Collect data in DataFrame
# for lake in data.keys():
    # LakeGroupFound = None
    # for LakeGroup in LakeGroups.keys():
        # for namePart in LakeGroups[LakeGroup]['names']:
            # if namePart in lake:
                # LakeGroupFound = LakeGroup
                # break
        # if LakeGroupFound is not None:
            # break
    # if LakeGroupFound is not None:
        # for t_idx in range(Nt):
            # if not np.isnan(data[lake][t_idx]).any():
                # ind = data[lake][t_idx]

                # lake_id = lake_ids.data[t_idx, ind[1], ind[0]]

                # lake_area = lakeData['areas'][t_idx][lake_id]/(1000**2)
                # lake_vol  = lakeData['volumes'][t_idx][lake_id]/(1000**3)

                # LakeGroups[LakeGroup]['area'][t_idx] += lake_area
                # LakeGroups[LakeGroup]['volume'][t_idx] += lake_vol

# #Save data in Textfile
# with open(tableGroupName, "w") as tab:
    # tab.write("Name\tYear[kaBP]\tArea[km^2]\tVolume[km^3]\n")

    # for LakeGroup in LakeGroups.keys():
        # for t_idx in range(Nt):
            # if LakeGroups[LakeGroup]['area'][t_idx] > 0:
                # #print(LakeGroup+"- "+t_name[t_idx]+": area:"+str(LakeGroups[LakeGroup]['area'][t_idx])+" vol:"+str(LakeGroups[LakeGroup]['volume'][t_idx]))
                # tab.write('{}\t{}\t{}\t{}\n'.format(LakeGroup, t[t_idx], LakeGroups[LakeGroup]['area'][t_idx], LakeGroups[LakeGroup]['volume'][t_idx]))

# ####Set Latex header format
# def fnum(x, n=2):
    # if type(x) is str:
        # return x
    # else:
        # if np.isnan(x):
            # return 'NaN'
        # else:
            # return ('$ {:,.'+str(n)+'f} $').format(x).replace(',', '\\,')

# fnum0 = lambda x: fnum(x,0)
# fnum1 = lambda x: fnum(x,1)
# fnum2 = lambda x: fnum(x,2)

# all_tables=list()

# headers = {'Name':'Name','kaBP': 'Time [$\\unit{\\mathrm{kaBP}}$]' ,'Area':'Area [$\\unit{\\mathrm{km}^2}$]', 'Volume':'Volume [$\\unit{\\mathrm{km}^3}$]', 'Level':'Level [$\\unit{\\mathrm{m}}$]', 'max. Depth': 'max. Depth [$\\unit{\\mathrm{m}}$]', 'Sink': 'Sink'}
# index_names = {'kaBP': 'Time [$\\unit{\\mathrm{kaBP}}$]', 'Name': 'Lake'}
# formatters={'kaBP':fnum0 ,'Area':fnum0, 'Volume':fnum1, 'Level':fnum1, 'max. Depth': fnum1}

# #Export as LaTex supertabular
# latex_table_long =  os.path.join(path_data,'latex', Name_prefix+"_long_table.tex")

# columns=[ 'Area', 'Volume', 'Level', 'max. Depth', 'Sink']
# indexes = ['kaBP', 'Name']

# header = [index_names[X] for X in indexes]
# header.extend([headers[X] for X in columns])

# df_filtered = LakesDF.sort_values(['kaBP' ,'Area'], ascending = [False, False])
# df_filtered.set_index(indexes, inplace=True)

# column_format = 'rl'+(len(header) -3)*'r'+'l'

# long_table_caption = "\\label{Tab:all_lakes_table}\n  Area, volume, lake level, maximum depth, and sink for all identified lakes within this study."

# save_tabular(latex_table_long, df_filtered, long_table_caption, formatters, column_format, columns, header, supertabular=True)
# all_tables.append(latex_table_long)

# #Export Lake as LaTex tabular
# columns=[ 'Area', 'Volume', 'Level', 'max. Depth', 'Sink']
# indexes = ['kaBP']

# header = [index_names[X] for X in indexes]
# header.extend([headers[X] for X in columns])
# column_format = 'r'+(len(header) -2)*'r'+'l'

# for lake_name in data.keys():
    # latex_table_lake =  os.path.join(path_data,'latex', Name_prefix+"_"+clean_file_name(lake_name)+"_table.tex")

    # LakeDF_filtered = LakesDF[LakesDF['Name'] == lake_name]
    # df_filtered = LakeDF_filtered.sort_values(['kaBP' ,'Area'], ascending = [True, False])
    # df_filtered.set_index(indexes, inplace=True)

    # table_caption = lake_name

    # save_tabular(latex_table_lake, df_filtered, table_caption, formatters, column_format, columns, header, supertabular=False)
    # all_tables.append(latex_table_lake)




# #Export Lake Groups
# columns=[ 'Area', 'Volume', 'Level', 'max. Depth', 'Sink']
# indexes = ['Name']

# header = [index_names[X] for X in indexes]
# header.extend([headers[X] for X in columns])
# column_format = 'l'+(len(header) -2)*'r'+'l'

# for LakeGroup in LakeGroups:
    # append = False
    # latex_table_lakegroup = os.path.join(path_data,'latex', Name_prefix+"_"+clean_file_name(LakeGroup)+"Group_table.tex")

    # for t_idx in range(Nt):
        # if LakeGroups[LakeGroup]['area'][t_idx] > 0:
            # t_cur = t[t_idx]
            # area_cur = LakeGroups[LakeGroup]['area'][t_idx]
            # vol_cur  = LakeGroups[LakeGroup]['volume'][t_idx]

            # table_caption = LakeGroup+' '+str(t_cur)+'kaBP'

            # LakeDF_filtered = LakesDF[[any([NP in LN for NP in LakeGroups[LakeGroup]['names']]) for LN in LakesDF['Name']] & (LakesDF['kaBP'] == t_cur)]
            # df_filtered = LakeDF_filtered.sort_values(['Area'], ascending = [False])
            # df_filtered = df_filtered.append({'Name':'\\middlehline \\textbf{Total}', 'Area': area_cur, 'Volume': vol_cur, 'kaBP':t_cur}, ignore_index=True)
            # df_filtered.set_index(indexes, inplace=True)

            # save_tabular(latex_table_lakegroup, df_filtered, table_caption, formatters, column_format, columns, header, supertabular=False, append=append, na_rep='')
            # append = True

    # all_tables.append(latex_table_lakegroup)


# all_tables_tex="\
# \documentclass[esurf, manuscript]{copernicus} \n\
# \usepackage{multirow} \n\
# \usepackage{supertabular} \n\
# \n\
# \\begin{document} \n\n\
# "+" \n\n".join(["\\input{"+X+"}" for X in all_tables])+" \n \n\
# \\end{document}"

# all_tables_out_tex = os.path.join(path_data,'latex', Name_prefix+"_all_tables.tex")
# with open(all_tables_out_tex, 'w') as fOut:
    # fOut.write(all_tables_tex)
    


    
# columns=['Area', 'Volume', 'max. Depth', 'Sink']
# indexes = ['kaBP', 'Name']

# header = [index_names[X] for X in indexes]
# header.extend([headers[X] for X in columns])
# column_format = 'rl'+(len(header) -3)*'r'+'l'

# latex_table_CN_YD =  os.path.join(path_data,'latex', Name_prefix+"_CN_YD_table.tex")

# lake_names = ['Agassiz', 'McConnell', 'McKenzie', 'Souris', 'Hind', 'Great Bear', 'Saskatchewan', 'Ojibway', 'Ojibway 2', 'Meadow'] #'Michigan', 'Erie-Huron', 'Ontario', 'Huron-Michigan', 'Erie',
# year_list = [15,14,13,12,11,10,9]

# LakeDF_filtered = LakesDF[[X&Y for (X,Y) in zip([N in lake_names for N in LakesDF['Name']], [Y in year_list for Y in LakesDF['kaBP']])]]
# df_filtered = LakeDF_filtered.sort_values(['kaBP', 'Area'], ascending = [False, False])
# df_filtered.set_index(indexes, inplace=True)

# table_caption = "Test"

# save_tabular(latex_table_CN_YD, df_filtered, table_caption, formatters, column_format, columns, header, supertabular=False)
# all_tables.append(latex_table_lake)
