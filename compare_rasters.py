import logging
import logging.config
import os
import pathlib

import yaml
import numpy as np
import rasterio

from config import (
    LOGGCONFIG,
    OSMRASTER,
    TERRADIR,
    WC_COMP_PATH,
    WC_OSM_COMP_PATH,
    OSM_COMP_PATH,
)


def get_tileyear(file: pathlib.Path) -> str:
    year = file.stem[-21:-17]
    return year


def get_tilename(file: pathlib.Path) -> str:
    name = file.stem[-11:-4]
    return name


def get_raster_to_compare(raster, datapath):
    for file in datapath.rglob("*_Map.tif"):
        year = get_tileyear(file)
        tile = get_tilename(file)
        if year == "2020" and tile == get_tilename(raster):
            return file
    logger.warning(f"Could not find corresponding WorldCover Raster for File {raster}. Cannot compare WC Data")
    return None


def get_rasterdata(rasterpath, comparepath):
    with rasterio.open(rasterpath) as raster:
        rasterdata = raster.read()
    with rasterio.open(comparepath) as raster:
        comparedata = raster.read()

    return rasterdata, comparedata


def save_raster(data, path, crs, transform):
    new_dataset = rasterio.open(
        path, 'w',
        driver='GTiff',
        height=data.shape[1],
        width=data.shape[2],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        compress='lzw'
    )
    new_dataset.write(data)
    new_dataset.close()


def compare_tiles(rasterpath, comparepath, resultdir):
    rasterdata, comparedata = get_rasterdata(rasterpath, comparepath)
    binary_change = np.equal(rasterdata, comparedata)
    resultpath = resultdir + f"{get_tilename(rasterpath)}_binary_change.tif"
    with rasterio.open(rasterpath) as raster:
        save_raster(binary_change.astype(np.uint8), resultpath, raster.crs, raster.transform)

    del rasterdata, comparedata, binary_change


def main(compare_wc=True, compare_wc_osm=True):
    wc_datapath = pathlib.Path(TERRADIR + "Maps/")
    for rasterpath in wc_datapath.rglob("*_Map.tif"):
        if get_tileyear(rasterpath) == "2021" and compare_wc:
            comparepath = get_raster_to_compare(rasterpath, wc_datapath)
            if comparepath is not None:
                resultdir = WC_COMP_PATH
                if not os.path.exists(resultdir):
                    os.makedirs(resultdir)
                compare_tiles(rasterpath, comparepath, resultdir)



        #TODO: add OSM comparison


if __name__ == "__main__":
    # warnings.filterwarnings("error")
    with open(LOGGCONFIG, "r") as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    logger = logging.getLogger(__name__)

    main()

    #TODO:
    # add logging
    # add typehint