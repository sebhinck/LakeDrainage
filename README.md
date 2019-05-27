# LakeDrainage
This tool is used to post-process data from the LakeCC model.

## Installation
To install run `make all` in this folder. Package is not installed globally; only here in this folder.

## Preparation
To apply this tool all results of all time-slices of the LakeCC model need to be merged into a single netCDF file.
Names and folders in the headers of `computeLakeCCDrainageData.py` and`exportLakeDrainageData.py` need to be adapted.
Check if projection matches `epsg` string in `exportLakeDrainageData.py`. Basins in `Basins.nc` need to match with basins defined in `c++/LakeBasins.hh` and `exportLakeDrainageData.py`.
Furthermore check lake locations and names in `exportLakeDrainageData.py`.

## Application
Run the script `computeLakeCCDrainageData.py` to label lake basins calculated from by the LakeCC and collect data about all lakes. This data is written to a netCDF file and two **pickle** Files.

After the first step run the script `exportLakeDrainageData.py` to process the identified lakes. The drainage route is determined and written to shape files. For these lakes several latex tales are created and written, that contain specified data.
