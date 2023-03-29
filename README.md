# Masterthesis

## Filter Setting:
### Filter Format
Needs to be xyz. \
Divided by ```,```, not ```=```

### Buffer Setting 
Width of lines must be the total width the line should be, not per Lane of road or radius.
The required Unit is Meters.

## Troubleshooting:
### Poetry Installation
If Poetry fails to install with the following error message:
```
Retrieved digest for link 
terracatalogueclient-0.1.14-py3-none-any.whl(md5:fe76adc43b984d5a4d0d6edc608eeff1) 
not in poetry.lock metadata 
['sha256:ae7c490a60d6bd23eab927f453d2fb0a9d3c3a7a4d208ab27bfdc5d77766c208']
```
run 
```
poetry config experimental.new-installer false
```
to use the old (warning: not recommended) installer and try again