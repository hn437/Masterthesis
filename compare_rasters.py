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


def get_osmyear(file: pathlib.Path) -> str:
    year = file.stem[-4:]
    return year


def get_osmtile(file: pathlib.Path) -> str:
    name = file.stem[:-5]
    return name


def get_wc_to_compare(raster, datapath):
    for file in datapath.rglob("*_Map.tif"):
        year = get_tileyear(file)
        tile = get_tilename(file)
        if year == "2020" and tile == get_tilename(raster):
            return file
    logger.warning(f"Could not find corresponding WorldCover Raster for WC File {raster}. Cannot compare WC Data")
    return None


def get_other_to_compare(raster, datapath):
    for file in datapath.rglob("*.tif"):
        year = get_tileyear(raster)
        tile = get_tilename(raster)
        if year == get_osmyear(file) and tile == get_osmtile(file):
            return file
    logger.warning(f"Could not find corresponding OSM Raster for WC File {raster}. Cannot compare Data between OSM and WC")
    return None


def get_osm_to_compare(raster, datapath):
    for file in datapath.rglob("*.tif"):
        year = get_osmyear(file)
        tile = get_osmtile(file)
        if year == "2020" and tile == get_osmtile(raster):
            return file
    logger.warning(f"Could not find corresponding OSM Raster for OSM File {raster}. Cannot compare OSM Data")
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


def detect_equality(rasterpath, comparepath, resultfile):
    rasterdata, comparedata = get_rasterdata(rasterpath, comparepath)
    #binary_change = np.equal(rasterdata, comparedata)
    # use isclose with a tolerance of 0 as it can compare nan values (both nan -> no change)
    binary_change = np.isclose(rasterdata, comparedata, rtol=0, atol=0, equal_nan=True)
    with rasterio.open(rasterpath) as raster:
        save_raster(binary_change.astype(np.uint8), resultfile, raster.crs, raster.transform)
    logger.info(f"Wrote binary change raster {resultfile.name}")

    del rasterdata, comparedata, binary_change


def detect_loss_of_nature(rasterpath, comparepath, resultfile):
    """
    Get "old" classification where raster was not classified as built up but is now.
    :param rasterpath:
    :param comparepath:
    :param resultfile:
    :return: None
    """
    rasterdata, comparedata = get_rasterdata(rasterpath, comparepath)
    outputdata = np.where((rasterdata==50) & (comparedata!=50),comparedata,0)
    with rasterio.open(rasterpath) as raster:
        save_raster(outputdata, resultfile, raster.crs, raster.transform)
    logger.info(f"Wrote raster indicating change of class to built up called {resultfile.name}")

    del rasterdata, comparedata, outputdata


def main(compare_wc=True, compare_wc_osm=True, compare_osm=True):
    wc_datapath = pathlib.Path(TERRADIR + "Maps/")
    osm_datapath = pathlib.Path(OSMRASTER)
    for rasterpath in wc_datapath.rglob("*_Map.tif"):
        if get_tileyear(rasterpath) == "2021" and compare_wc:
            comparepath = get_wc_to_compare(rasterpath, wc_datapath)
            if comparepath is not None:
                resultdir = WC_COMP_PATH
                if not os.path.exists(resultdir):
                    os.makedirs(resultdir)
                # binary change
                resultfile = pathlib.Path(resultdir + f"{get_tilename(rasterpath)}_wc_binary_change.tif")
                detect_equality(rasterpath, comparepath, resultfile)
                # old classes have become built up
                resultfile = pathlib.Path(resultdir + f"{get_tilename(rasterpath)}_wc_loss_of_nature.tif")
                detect_loss_of_nature(rasterpath, comparepath, resultfile)
        if compare_wc_osm:
            comparepath = get_other_to_compare(rasterpath, osm_datapath)
            if comparepath is not None:
                resultdir = WC_OSM_COMP_PATH
                if not os.path.exists(resultdir):
                    os.makedirs(resultdir)
                resultfile = pathlib.Path(resultdir + f"{get_tilename(rasterpath)}_{get_tileyear(rasterpath)}_wc_osm_binary_change.tif")
                detect_equality(rasterpath, comparepath, resultfile)
    for rasterpath in osm_datapath.rglob("*.tif"):
        if get_osmyear(rasterpath) == "2021" and compare_osm:
            comparepath = get_osm_to_compare(rasterpath, osm_datapath)
            if comparepath is not None:
                resultdir = OSM_COMP_PATH
                if not os.path.exists(resultdir):
                    os.makedirs(resultdir)
                # binary change
                resultfile = pathlib.Path(resultdir + f"{get_osmtile(rasterpath)}_osm_binary_change.tif")
                detect_equality(rasterpath, comparepath, resultfile)
                # old classes have become built up
                resultfile = pathlib.Path(resultdir + f"{get_osmtile(rasterpath)}_osm_loss_of_nature.tif")
                detect_loss_of_nature(rasterpath, comparepath, resultfile)

        #TODO: find specific change instead of binary (equal=1, not_equal=0)


if __name__ == "__main__":
    # warnings.filterwarnings("error")
    with open(LOGGCONFIG, "r") as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    logger = logging.getLogger(__name__)

    main(compare_wc=False, compare_wc_osm=False, compare_osm=True)

    #TODO:
    # add logging
    # add typehint