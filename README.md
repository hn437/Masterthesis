# Masterthesis
This repository contains the code used for the Masterthesis of the author.\
The repository also contains additional data from the appendix of the thesis. This data 
can be found in folder [Appendix](./appendix).

The thesis attempts to assess the temporal quality of OpenStreetMap (OSM) data in 
relation to land use/land cover (LULC) information. This is done using the example of 
the expansion of built-up areas. For this purpose, LULC maps are created for the Areas 
of Interest (AoI) from OSM data at two different points in time, which follow the class 
definitions of the WorldCover (WC) data. The AoI sections of the WC data are downloaded 
for the same two points in time to be used as reference data. The OSM-based maps 
are evaluated for completeness and accuracy in comparison to the WC data. It is also 
determined to what percentage an expansion of the development corresponds in the 
respective datasets. 
In addition, the calculation of the mapping saturation indicator of the Ohsome Quality 
API is used to calculate how completely the respective LULC classes are mapped in the 
OSM maps and a correlation analysis is carried out between the characteristics of the 
AoI and the correspondence of the development expansion.


## Usage of the Code
In order to run the Code, the required environment must be set up. This can be done by 
installing it from the [pyproject.toml](./pyproject.toml) file with Poetry. Furthermore,
the user must set the configurations in the [config.py](./config.py) file. The code can 
be run by executing the [run_full_analysis.py](./run_full_analysis.py) file.

## Requirements
### General Requirements
- Python >=3.9, <=3.11
- Poetry installed as Python Package Manager
- User Account for Terrascope Platform
- Adapted Version of the 
[Ohsome Quality API](https://github.com/GIScience/ohsome-quality-api)
(see section 'Mapping Saturation Indicator Calculation' for further information)


### AoI Specification
Up to 999 Inputfiles in the GeoJSON Format can be specified. Each file can hold multiple
Polygons. Those must have an attribute 'id' with values from 0 to 999 allowed. 
Furthermore, in order to be able to run the correlation analysis, the each polygon must 
have the following attributes assigned: "Population" (int), "X-Coordinate" (float), 
"Y-Coordinate"(float), "University" (boolean), and "Disposable Income" (int).


### Mapping Saturation Indicator Calculation
As the filters defined in this repository are not default topics in the ohsome quality 
API (OQAPI), the script [calculate_oqapi.py](./calculate_oqapi.py) does not work with 
public OQAPI. Therefore, it is recommended to set up a local OQAPI instance, define the 
filters used in this analysis as topics in OQAPI, and use it for the calculation. For 
instructions on how to set up a local OQAPI instance, see the 
[Ohsome Quality API Repository](https://github.com/GIScience/ohsome-quality-api). Each 
filter must be defined as a topic in OQAPI. Fur further information on OQAPI topics, 
check out the 
[topic documentation of OQAPI](https://github.com/GIScience/ohsome-quality-api/blob/main/docs/topic.md).


## Workflow Adaption Options 
### Filter Setting
In order to define which OSM Features should be considered per LULC class, a filter was 
defined per class. These filters can be adapted. However, the filter format must be
kept. The filters are stored in the [filter](./data/filter) directory.

#### Filter Format
The first line of the file contains the OSM tags which should be used to query Polygons, 
the second line contains the OSM tags which should be used to query line features. Each 
file must not have more than 2 lines. If only line features should be queried, still 
write them to second line.

#### Buffer Setting 
The file [buffersizes.txt](./data/buffersizes.txt) contains the width with which line 
features should be buffered. Each line must contain a Key-Value-Pair and the Buffer 
width. The tag and width must be divided by ```,```, not ```=```.\
The Width of lines must be the stated as total width the line should be, not per Lane.
The required Unit is Meters.

#### Confidence Level Setting
The file [keyconfidences.txt](./data/keyconfidences.txt) contains the confidence value 
which will be assigned to each queried OSM feature based on the key used to query the 
respective feature. When adapting the filter definitions, this file must be updated as 
well. The file must contain an OSM key per line, paired with the to be associated level 
of confidence. The key and confidence level must be divided by ```,```, not ```=```.\
The confidence level must be an integer between 1 and 4. It defines the confidence that 
features tagged with the respective key are associated with the correct LULC class. It 
is used to define which features will be overwritten by other in the final map if 
features overlay each other. The higher the confidence level, the less likely the 
feature will be overwritten by another feature and consequently the more likely the 
feature will be represented in the final map.


## Troubleshooting
### Poetry Installation
If Poetry fails to install the dependencies with the following error message:
```
Retrieved digest for link 
terracatalogueclient-0.1.14-py3-none-any.whl(md5:xxxxx) 
not in poetry.lock metadata 
['sha256:xxxxx']
```
run 
```
poetry config experimental.new-installer false
```
to use the old (warning: not recommended) installer and try again