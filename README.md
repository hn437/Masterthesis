# Masterthesis

## General Function
Takes Area of Interest, downloads Terradata for both years for it, downloads OSM data and processes it.
As terradata is downloaded for both years and OSM script iterates over downloaded maps, 
no need to implement processing for both years in that script again

## AoI Specification
Up to 999 Inputfiles can be specified. Each file can hold multiple Polygons. Those must 
have an attribute 'id' with values from 0 to 999 allowed

## Filter Setting:
### Filter Format
First line filters that should be used for Poly, second for line. Must not have more 
than 2 lines. if line features only, still write them to second line.

### Buffer Setting 
Key-Value-Pair and Buffer value must be divided by ```,```, not ```=```.\
Width of lines must be the total width the line should be, not per Lane of road or radius.
The required Unit is Meters.

### Mapping Saturation Calculation
As the filters defined in this repository are not default topics in the ohsome quality 
API, the script ```calculate_oqapi.py``` does not work with public OQAPI. Therefore, it
is recommended to set up a local OQAPI instance, define the Filters of this repository
as topics, and use it for the calculation. For instructions on how to set up a local 
OQAPI instance, see the 
[Ohsome Quality API Repository](https://github.com/GIScience/ohsome-quality-api).


## Troubleshooting:
### Poetry Installation
If Poetry fails to install with the following error message:
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