import logging
import logging.config
import os
import pathlib
import shutil
import sys

import geopandas as gpd
import rasterio as rio
import shapely
import yaml
from rasterio.mask import mask
from rasterio.merge import merge
from shapely.geometry import box
from terracatalogueclient import Catalogue

from config import INFILES, INPUTDIR, LOGGCONFIG, PASSWORD, TERRADIR, USERNAME


def import_geodata(input_dir: str, infile: str) -> gpd.GeoDataFrame:
    if infile is not None:
        path_to_file = pathlib.Path(input_dir, infile)
        df = gpd.read_file(path_to_file)

        if df.crs != 4326:
            df = df.to_crs(4326)

    return df


def download_terrascope_data(directory: pathlib.Path) -> None:
    # Authenticate to the Terrascope platform and create catalogue object
    catalogue = Catalogue().authenticate_non_interactive(
        username=USERNAME, password=PASSWORD
    )

    if len(INFILES) == 0:
        for filename in os.listdir(INPUTDIR):
            if filename.endswith(".geojson"):
                INFILES.append(filename)
    if len(INFILES) > 999:
        logger.error("Too many Files stated for Input. Max 999 Files allowed")
        sys.exit(1)

    if not os.path.exists(directory):
        os.makedirs(directory)

    file_counter = 0
    for infile in INFILES:
        logger.info(f"Working on Inputfile {file_counter +1} of {len(INFILES)}")
        # import geodata of extent to be imported from WorldCover
        gdf = import_geodata(INPUTDIR, infile)

        if len(gdf.index) > 999:
            logger.error(
                f"Too many Features in File {infile}. Max 999 Features allowed"
            )
            continue

        for index in gdf.index:
            logger.info(f"Working on feature {index+1} of {len(gdf)}")
            feature = gdf.loc[[index]]
            geom = feature.geometry[index]

            products_2020 = catalogue.get_products(
                "urn:eop:VITO:ESA_WorldCover_10m_2020_V1", geometry=geom
            )
            products_2021 = catalogue.get_products(
                "urn:eop:VITO:ESA_WorldCover_10m_2021_V2", geometry=geom
            )

            # download the products to the given directory
            logger.info(f"Started download of WorldCover data")
            catalogue.download_products(products_2020, directory, force=True)
            catalogue.download_products(products_2021, directory, force=True)
            logger.info(f"Finished download of WorldCover data")

            clean_terradata(directory, file_counter, index, geom)

        file_counter += 1

    # remove scratch dir
    shutil.rmtree(directory)
    logger.info(f"Finished downloading WorldCover Data for all Files and Features\n\n")


def clean_terradata(
    scratch_dir: pathlib.Path,
    file_no,
    feature_id: int,
    feature_geom: shapely.geometry.polygon.Polygon,
) -> None:
    maps_dir = TERRADIR + "Maps/"
    quality_dir = TERRADIR + "Quality/"
    if not os.path.exists(maps_dir):
        os.makedirs(maps_dir)
    if not os.path.exists(quality_dir):
        os.makedirs(quality_dir)

    # clip all raster
    logger.info(f"Clipping WorldCover Data to Feature Extent")
    clip_all_downloads(scratch_dir, feature_geom)

    # check if AoI overlays multiple WorldCover Tiles
    if len(list(scratch_dir.rglob("*2020*_Map.tif"))) > 1:
        logger.info(f"Merging WorldCover Data and saving it")
        # make lists of files olding Maps or Data Quality per year
        maps_20 = list(scratch_dir.rglob("*2020*_Map.tif"))
        maps_21 = list(scratch_dir.rglob("*2021*_Map.tif"))
        quality_20 = list(scratch_dir.rglob("*2020*_InputQuality.tif"))
        quality_21 = list(scratch_dir.rglob("*2021*_InputQuality.tif"))

        # Merge tiles per year and type
        merge_tiles(maps_20, maps_dir, "Map", file_no, feature_id)
        merge_tiles(quality_20, quality_dir, "InputQuality", 1, feature_id)
        merge_tiles(maps_21, maps_dir, "Map", file_no, feature_id)
        merge_tiles(quality_21, quality_dir, "InputQuality", 1, feature_id)

    else:
        logger.info(f"Saving WorldCover Data")
        # AoI within a single raster tile -> no merging, but renaming and moving to dir
        for file in scratch_dir.rglob("*_Map.tif"):
            new_name = f"ESA_WorldCover_10m_{str(file.stem)[19:29]}f{file_no:03}id{feature_id:03}_Map.tif"
            os.rename(file, maps_dir + new_name)

        for file in scratch_dir.rglob("*_InputQuality.tif"):
            new_name = f"ESA_WorldCover_10m_{str(file.stem)[19:29]}f{file_no:03}id{feature_id:03}_InputQuality.tif"
            os.rename(file, quality_dir + new_name)


def clip_all_downloads(
    scratch_dir: pathlib.Path, feature_geom: shapely.geometry.polygon.Polygon
) -> None:
    # create bounding box instead of using original feature
    bounds = feature_geom.bounds
    geom = box(*bounds)

    for tile in list(scratch_dir.rglob("*.tif")):
        with rio.open(tile) as src:
            out_image, out_transform = mask(src, [geom], crop=True, all_touched=True)
            out_meta = src.meta.copy()  # copy the metadata of the source DEM

        out_meta.update(
            {
                "driver": "Gtiff",
                "height": out_image.shape[1],  # height starts with shape[1]
                "width": out_image.shape[2],  # width starts with shape[2]
                "transform": out_transform,
            }
        )

        out_name = str(tile.stem)[:4] + "FileMasked" + str(tile.name)[14:]
        out_path = pathlib.Path(scratch_dir / out_name)
        with rio.open(out_path, "w", **out_meta) as dst:
            dst.write(out_image)
        os.remove(tile)

    for file in os.listdir(scratch_dir):
        d = os.path.join(scratch_dir, file)
        if os.path.isdir(d):
            os.rmdir(d)


def merge_tiles(
    tiles: list, dir_path: str, tiletype: str, file_no: int, feature_id: int
) -> None:
    raster_to_mosiac = []
    for tile in tiles:
        raster = rio.open(tile)
        raster_to_mosiac.append(raster)

    mosaic, output = merge(raster_to_mosiac)

    output_meta = raster.meta.copy()
    output_meta.update(
        {
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": output,
        }
    )
    output_path = (
        dir_path
        + f"ESA_WorldCover_10m_{str(tile.stem)[19:28]}_f{file_no:03}id{feature_id:03}_{tiletype}.tif"
    )
    with rio.open(output_path, "w", **output_meta) as m:
        m.write(mosaic)

    for file in tiles:
        os.remove(file)


def main():
    path_terradir_scratch = pathlib.Path(TERRADIR + "scratch/")

    download_terrascope_data(path_terradir_scratch)

    # TODO:
    #  Improve Logging
    #  add Docstrings
    #  add Comments


if __name__ == "__main__":
    with open(LOGGCONFIG, "r") as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    logger = logging.getLogger(__name__)

    main()
