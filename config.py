# configurations needed for the scripts

## General Settings
LOGGCONFIG = "./data/logging.yml"

## configs for script download_worldcover_data.py
INPUTDIR = "./data/input/"
INFILES = [
    "infile.geojson"
]  # leave empty to use all .geojson files in input dir
TERRADIR = "./data/terradata/"

## configs for script download_osm_data.py
APIENDPOINT = "https://api.ohsome.org/v1/elements/geometry"
TIME20 = "2021-01-01"
TIME21 = "2022-01-01"
FILTERPATH = "./data/filter/"
CONFICENCEDICT = "./data/keyconfidences.txt"
BUFFERSIZES = "./data/buffersizes.txt"
CLASSCODES = "./data/classcodes.txt"
OSMRASTER = "./data/osmraster/"

## configs for comparison
COMP_PATH = "./data/comparison/"
WC_COMP_PATH = COMP_PATH + "WC/"
WC_OSM_COMP_PATH = COMP_PATH + "WC_OSM/"
OSM_COMP_PATH = COMP_PATH + "OSM/"
CM_PATH = COMP_PATH + "CM/"
MAJORITY_SIZE = 5

## configs for ohsome quality API
base_url = "http://127.0.0.1:8080"
endpoint = "/indicators"
indicator = "/mapping-saturation"
URL = base_url + endpoint + indicator

HEADERS = {"accept": "application/json"}


## Terrascope credentials
NO_OF_RETRIES = 3
USERNAME = ""
PASSWORD = ""
