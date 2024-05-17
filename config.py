# configurations needed for the scripts

## General Settings
# get the path to the logging configuration file
LOGGCONFIG = "./data/logging.yml"


## configs for script download_worldcover_data.py
# path to the directory where the input/AoI data is stored
INPUTDIR = "./data/input/"
# which files in the input dir should be used as AoIs? Set empty list to use all
# .geojson files in input dir
INFILES = []
# path to the directory where the WC data should be stored
TERRADIR = "./data/terradata/"


## configs for script download_osm_data.py
# set API endpoint of the ohsome API used to query the OSM Features
APIENDPOINT = "https://api.ohsome.org/v1/elements/geometry"
# set the dates for which the OSM data should be queried
TIME20 = "2021-01-01"
TIME21 = "2022-01-01"
# set the path to the directory where the filter definitions (which Features to be
# queried per class) are stored
FILTERPATH = "./data/filter/"
# text file to be red in as dictionary defining the confidence values per OSM key
CONFICENCEDICT = "./data/keyconfidences.txt"
# text file to be red in as dictionary defining buffer widths for OSM line features
BUFFERSIZES = "./data/buffersizes.txt"
# text file to be red in as dictionary defining the class codes for LULC classes
CLASSCODES = "./data/classcodes.txt"
# set the path to the directory where the OSM data should be stored
OSMRASTER = "./data/osmraster/"


## configs for comparison
# set paths were resulting data of raster comparisons should be stored
# parent directory for all comparison results
COMP_PATH = "./data/comparison/"
# subdirectories for the different comparison results
# WC comparison over the years
WC_COMP_PATH = COMP_PATH + "WC/"
# WC vs OSM comparison per year
WC_OSM_COMP_PATH = COMP_PATH + "WC_OSM/"
# OSM comparison over the years
OSM_COMP_PATH = COMP_PATH + "OSM/"
# dir where confusion matrices should be stored
CM_PATH = COMP_PATH + "CM/"


## configs for ohsome quality API
# set base URL where to reach the ohsome quality API
base_url = "http://127.0.0.1:8080"
# set the endpoint of the API
endpoint = "/indicators"
# set the indicator to be queried
indicator = "/mapping-saturation"
# set the full URL to be used for the query
URL = base_url + endpoint + indicator
# set the headers for the API request
HEADERS = {"accept": "application/json"}


## Terrascope credentials
# define the number of retries if downloading from Terrascope fails
NO_OF_RETRIES = 3
# set your Terrascope username
USERNAME = ""
# set your Terrascope password
PASSWORD = ""