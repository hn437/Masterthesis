import asyncio
import logging

logger = logging.getLogger("__main__")
import logging.config
import os
import pathlib
from time import sleep
from typing import Coroutine, Optional

import geocube.exceptions
import geojson
import geopandas as gpd
import httpx
import pandas
import rasterio
import rioxarray
import yaml
from geocube.api.core import make_geocube
from geojson import Feature, FeatureCollection
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from shapely.geometry import box
from shapely.validation import make_valid

from config import (
    APIENDPOINT,
    BUFFERSIZES,
    CLASSCODES,
    CONFICENCEDICT,
    FILTERPATH,
    LOGGCONFIG,
    NO_OF_RETRIES,
    OSMRASTER,
    TERRADIR,
    TIME20,
    TIME21,
)


def get_tilename(file: pathlib.Path) -> str:
    name = file.stem[-13:-4]
    return name


def get_tileyear(file: pathlib.Path) -> str:
    year = file.stem[-23:-19]
    return year


def get_extent(file: pathlib.Path) -> FeatureCollection:
    with rasterio.open(file) as raster:
        bounds = raster.bounds
    geom = box(*bounds)
    feature_collection = FeatureCollection([Feature(geometry=geom)])
    return feature_collection


def load_dict(path: str, val_type: type) -> dict:
    d = {}
    with open(path) as f:
        for line in f:
            (key, val) = line.split(",")
            d[key] = val_type(val)

    return d


def fixing_geometry(geometry):
    if not geometry.is_valid:
        return make_valid(geometry)
    else:
        return geometry


async def get_vector_areas(
    path_to_filter: pathlib.Path,
    time: str,
    extent: FeatureCollection,
    confidence_dict: dict,
    buffer_dict: dict,
    class_codes: dict,
) -> Optional[gpd.GeoDataFrame]:
    logger.info(f"Querying OSM Data for Filter {path_to_filter.stem}")

    df_of_features = gpd.GeoDataFrame()
    buffered_linefeatures = gpd.GeoDataFrame()

    with open(path_to_filter) as f:
        lines = f.readlines()
    if len(lines) < 1:
        logger.critical(
            f"Filter {path_to_filter.stem} empty! Fix Filter before attempting to download OSM data"
        )
        exit()

    counter = 0

    for try_no in range(NO_OF_RETRIES):
        try:
            for line in lines:
                osmfilter = line
                if len(osmfilter) != 0 and osmfilter != "\n" and counter == 0:
                    filterquery = f"({osmfilter}) and geometry:polygon"
                elif len(osmfilter) != 0 and osmfilter != "\n" and counter == 1:
                    filterquery = f"({osmfilter}) and geometry:line"
                else:
                    counter += 1
                    continue

                data = {
                    "bpolys": geojson.dumps(extent),
                    "time": time,
                    "filter": filterquery,
                    "properties": "tags",
                }
                # response = requests.post(APIENDPOINT, data=data)
                async with httpx.AsyncClient(timeout=None) as client:
                    response = await client.post(APIENDPOINT, data=data)
                response.raise_for_status()
                if response.status_code != 200:
                    logger.error(
                        f"Ohsome API Query for Filter {path_to_filter.stem} not successful. Status Code: {response.status_code}. {response.text}"
                    )
                    break
                else:
                    datapart = gpd.GeoDataFrame.from_features(
                        response.json()["features"]
                    )
                    del response
                    if len(datapart.index) > 0:
                        # only continue processing if features were found
                        # set crs as it is not returned in response, but is always WGS 84
                        datapart.set_crs(4326, inplace=True)
                        # dropping all columns (= OSM Keys) not actively queried
                        column_names = datapart.columns.values.tolist()[3:]
                        confidence_keys = [key for key in confidence_dict]
                        col_to_drop = list(set(column_names) - set(confidence_keys))
                        datapart = datapart.drop(columns=col_to_drop)

                        # if dealing with line features: query usable UTM projection and reproject data to be able to buffer them by meters
                        if counter == 1:
                            # get example coordinate to query useable UTM projection
                            tile_corner_coords = extent["features"][0]["geometry"][
                                "coordinates"
                            ][0][0]
                            # query UTM code for that coordinate
                            utm_crs_list = query_utm_crs_info(
                                datum_name="WGS 84",
                                area_of_interest=AreaOfInterest(
                                    west_lon_degree=tile_corner_coords[0],
                                    south_lat_degree=tile_corner_coords[1],
                                    east_lon_degree=tile_corner_coords[0],
                                    north_lat_degree=tile_corner_coords[1],
                                ),
                            )
                            utm_code = utm_crs_list[0].code
                            # reproject feature to queried UTM
                            datapart.to_crs(utm_code, inplace=True)

                        # iterate over features to assign confidence level and buffer the lines
                        for index in datapart.index:
                            row = datapart.loc[[index]]
                            # drop columns without values
                            row = row[row.columns[~row.isnull().all()]]
                            used_keys = row.columns.values.tolist()[3:]

                            if counter == 0:
                                # iterate over features to assign confidence level of polygons
                                if any(
                                    i in used_keys
                                    for i in [
                                        k for k, v in confidence_dict.items() if v == 4
                                    ]
                                ):
                                    datapart.at[index, "confidence"] = int(4)
                                elif any(
                                    i in used_keys
                                    for i in [
                                        k for k, v in confidence_dict.items() if v == 2
                                    ]
                                ):
                                    datapart.at[index, "confidence"] = int(2)
                                else:
                                    datapart.at[index, "confidence"] = int(1)
                            else:
                                # iterate over features to buffer the lines
                                buffer_dist = None
                                for key in used_keys:
                                    # get each key, get the value for this key of this feature anc check
                                    #  if it is in the dict of buffer values.
                                    combined_key = f"{key}={row[key][index]}"
                                    if combined_key in buffer_dict:
                                        buffer_dist = buffer_dict[combined_key]
                                        break
                                if buffer_dist is not None:
                                    # buffer feature. Divide by 2, as the input defines the buffer radius
                                    row["geometry"] = row.geometry.buffer(
                                        buffer_dist / 2
                                    )
                                    # reproject feature back to WGS 84 to be able to add them to polygon features
                                    row = row.to_crs(4326)
                                    # add feature to df of buffered features
                                    buffered_linefeatures = pandas.concat(
                                        [buffered_linefeatures, row], ignore_index=True
                                    )

                        if counter == 1:
                            # if features are a line features, write confidence level 2
                            if len(buffered_linefeatures.index) == 0:
                                # if line features have no buffer dist specified do not add them but warn
                                logger.warning(
                                    f"Some line features for filter '{path_to_filter.stem}' were queried, but no buffer size specified. Those features were ignored!"
                                )
                                continue
                            else:
                                datapart = buffered_linefeatures
                                del buffered_linefeatures
                                datapart["confidence"] = int(3)

                        df_of_features = pandas.concat(
                            [df_of_features, datapart], ignore_index=True
                        )
                    counter += 1

            df_of_features["class_code"] = int(class_codes[path_to_filter.stem])

            logger.info(f"Finished querying OSM Data for Filter {path_to_filter.stem}")
            return df_of_features
        except Exception:
            if try_no < NO_OF_RETRIES - 1:
                logger.warning(
                    f"Could not query data for Filter {path_to_filter.stem} at try No. {try_no + 1}. Retrying..."
                )
                # wait 10 seconds before retrying
                sleep(15)
                continue
            else:
                logging.exception(
                    f"Could not query data for Filter {path_to_filter.stem}."
                )
                return None


async def gather_with_semaphore(tasks: list, *args, **kwargs) -> Coroutine:
    """A wrapper around `gather` to limit the number of tasks executed at a time."""
    # Semaphore needs to be initiated inside the event loop
    semaphore = asyncio.Semaphore(3)

    async def sem_task(task):
        async with semaphore:
            return await task

    return await asyncio.gather(*(sem_task(task) for task in tasks), *args, **kwargs)


def resolve_overlays(input_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    logger.info("Cleaning DF")
    # drop duplicates
    gdf_without_duplicates = input_df.drop_duplicates(ignore_index=True)
    del input_df
    # attempt to fix invalid geometries
    gdf_without_duplicates.geometry = gdf_without_duplicates.geometry.apply(
        lambda geom: fixing_geometry(geom)
    )
    gdf_area = gdf_without_duplicates.explode(ignore_index=True)
    # drop geoms != Polygons
    polys = gdf_area[gdf_area.geom_type == "Polygon"]
    no_polys = gdf_area[gdf_area.geom_type != "Polygon"]
    if len(no_polys) > 0:
        logger.warning(
            f"Number of Features which are not Polygons: {len(no_polys)}. Types: {set(no_polys.geom_type.to_list())}"
        )

    # check for invalid geometries
    input_df = polys[polys.is_valid]

    logger.info("Attempting to resolve Overlay of Features")
    # check if overlays exist within gdf
    overlays_exist = False
    for index, row in input_df.iterrows():
        # TODO: Check if this may just slows down or if its actually improving
        # create temporary layer of all features except the one to be checked
        data_temp1 = input_df.drop(input_df.iloc[[index]].index)
        # overlap feature with all other features
        overlaps = data_temp1[data_temp1.geometry.overlaps(row.geometry)]
        # check if intersection occurred
        if len(overlaps) > 0:
            # if overlap occurred stop searching for overlaps and set variable accordingly
            overlays_exist = True
            break
    del data_temp1, overlaps, index, row
    if overlays_exist:
        # create a dataframe of all elements of the highest confidence level
        cleaned_features = input_df[input_df["confidence"] == 4]
        for confidence in range(1, 4):
            # get features of lower confidence value, get the difference to features of
            #  higher confidence levels (-> not overlaid by features of a higher
            #  confidence level) and this difference to the new dataframe
            df_base = input_df[input_df["confidence"] == confidence]
            if len(df_base) == 0:
                continue
            clip_features = input_df[input_df["confidence"] > confidence]
            res_difference = df_base.overlay(
                clip_features, how="difference", keep_geom_type=False
            )
            # explode to make polygons from multipolygons
            res_difference_single = res_difference.explode(ignore_index=True)
            # only use polygons, so drop line or point features which may resulted from overlay
            res_difference_polys = res_difference_single[
                res_difference_single.geom_type == "Polygon"
            ]

            cleaned_features = pandas.concat(
                [cleaned_features, res_difference_polys], ignore_index=True
            )
        return cleaned_features
    else:
        return input_df


async def query_osm_data(
    extent: FeatureCollection,
    confidence_dict: dict,
    buffer_dict: dict,
    class_codes: dict,
    time: str,
) -> Optional[gpd.GeoDataFrame]:
    logger.info("Querying OSM Data")
    # create task list, getting all vector features for each filter
    tasks = []
    for filter_class in pathlib.Path(FILTERPATH).rglob("*.txt"):
        tasks.append(
            get_vector_areas(
                filter_class, time, extent, confidence_dict, buffer_dict, class_codes
            )
        )

    # await the result for all ohsome queries
    tasks_results = await gather_with_semaphore(tasks, return_exceptions=True)
    # create empty gdf to collect all features
    if any(not isinstance(n, gpd.GeoDataFrame) for n in tasks_results):
        logger.error(f"Cannot Process Vector DataFrames due to Errors")
        for index, element in enumerate(tasks_results):
            if not isinstance(element, gpd.GeoDataFrame):
                logger.error(
                    f"Cannot process vector layers. Critical Element index: {index}. {element}"
                )
        return None
    else:
        # drop empty GeoDataFrames
        tasks_results = [df for df in tasks_results if not df.empty]
        logger.info(
            f"Successfully queried OSM Data and got Features for {len(tasks_results)} Classes"
        )
        all_vector_data = gpd.GeoDataFrame(
            pandas.concat(tasks_results, ignore_index=True), crs=4326
        )
        del tasks, tasks_results
        if len(all_vector_data.index) > 0:
            all_vector_data = resolve_overlays(all_vector_data)

            # TODO: remove below. For Testing only
            # with open("./data/test/gdf_cleaned.geojson", "w") as f:
            #    f.write(all_vector_data.to_json())

            return all_vector_data
        else:
            logger.warning(f"No OSM Data was found!")
            return None


def write_as_raster(df: gpd.GeoDataFrame, rastertile: pathlib.Path, time: str) -> None:
    # get WorldCover raster to make new raster with same properties
    logging.info(f"Attempting to write Raster from Vector Features")
    wc_data = rioxarray.open_rasterio(rastertile)
    try:
        osm_raster = make_geocube(
            vector_data=df, measurements=["class_code"], like=wc_data, fill=999
        ).astype("int16")
        del df
        tilename = get_tilename(rastertile)
        osm_raster_path = OSMRASTER + tilename + f"_{time}.tif"
        if not os.path.exists(OSMRASTER):
            os.makedirs(OSMRASTER)
        osm_raster.rio.to_raster(osm_raster_path)
    except geocube.exceptions.VectorDataError:
        logger.error(f"Cannot create Rastertile {rastertile.name} in year {time}.")

    del wc_data


async def main():
    # load dicts needed for all rasterfiles containing the class codes, key-confidences and buffer sizes
    confidence_dict = load_dict(CONFICENCEDICT, int)
    buffer_dict = load_dict(BUFFERSIZES, float)
    class_codes = load_dict(CLASSCODES, int)
    logger.info(f"Successfully loaded info dicts")

    for rastertile in pathlib.Path(TERRADIR + "Maps/").rglob("*_Map.tif"):
        logger.info(f"Started with {rastertile.name}")
        # Get the year represented by the rasterfile and set Ohsome download date accordingly
        year = get_tileyear(rastertile)
        if year == "2020":
            time = TIME20
        elif year == "2021":
            time = TIME21
        else:
            logger.warning(
                f"Could not derive year for rasterfile {rastertile.name}. Cannot process this raster!"
            )
            continue

        # get the extent of the raster for which OSM data should be downloaded
        bound_featurecol = get_extent(rastertile)
        # download and process all relevant vector data overlaying this raster
        osm_data = await query_osm_data(
            bound_featurecol, confidence_dict, buffer_dict, class_codes, time
        )
        if osm_data is not None:
            # convert vector data in raster data and save it
            write_as_raster(osm_data, rastertile, year)
            # TODO: Einfach keins, wie im moment?
            logger.info(f"Finished with {rastertile.name}")
        else:
            logger.warning(
                f"No OSM Data could be queried for Rastertile {rastertile.name} in year {time}\n"
            )

    # TODO:
    #  functionen weiter aufsplitten
    #  comments und docstrings adden


if __name__ == "__main__":
    # warnings.filterwarnings("error")
    with open(LOGGCONFIG, "r") as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    logger = logging.getLogger(__name__)

    asyncio.run(main())
