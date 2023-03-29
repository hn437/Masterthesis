# Masterthesis

## General Function
Takes Area of Interest, downloads Terradata for it, downloads OSM data and processes it

## Filter Setting:
### Filter Format
First line filters that should be used for Poly, second for line. Must not have more 
than 2 lines. if line only, still write them to second line.

### Buffer Setting 
Key-Value-Pair and Buffer value must be divided by ```,```, not ```=```.\
Width of lines must be the total width the line should be, not per Lane of road or radius.
The required Unit is Meters.

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