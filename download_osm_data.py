import os

import geocube.exceptions
import geojson
import geopandas as gpd
import pandas
import pathlib
import rasterio
import requests
import rioxarray
from geocube.api.core import make_geocube
from geojson import Feature, FeatureCollection
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from shapely.geometry import box

from config import (
    APIENDPOINT,
    BUFFERSIZES,
    CONFICENCEDICT,
    FILTERPATH,
    OSMRASTER,
    TERRADIR,
    TIME,
)


def get_tilename(file: str) -> str:
    name = file[-15:-8]
    return name


def get_extent(file: pathlib.Path) -> FeatureCollection:
    with rasterio.open(file) as raster:
        bounds = raster.bounds
    geom = box(*bounds)
    feature_collection = FeatureCollection([Feature(geometry=geom)])
    return feature_collection


def load_dict(path: str) -> dict:
    d = {}
    with open(path) as f:
        for line in f:
            (key, val) = line.split(",")
            d[key] = int(val)

    return d


def get_vector_areas(
    path_to_filter: pathlib.Path,
    time: str,
    extent: FeatureCollection,
    confidence_dict: dict,
    buffer_dict: dict,
) -> gpd.GeoDataFrame:
    df_of_features = gpd.GeoDataFrame()
    buffered_linefeatures = gpd.GeoDataFrame()

    with open(path_to_filter) as f:
        lines = f.readlines()

    counter = 0
    for line in lines:
        osmfilter = line
        if len(osmfilter) != 0 and counter == 0:
            filterquery = f"({osmfilter}) and geometry:polygon"
        elif len(osmfilter) != 0 and counter == 1:
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
        response = requests.post(APIENDPOINT, data=data)
        response.raise_for_status()

        datapart = gpd.GeoDataFrame.from_features(response.json()["features"])
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
            tile_corner_coords = extent["features"][0]["geometry"]["coordinates"][0][0]
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
                    for i in [k for k, v in confidence_dict.items() if v == 3]
                ):
                    datapart.at[index, "confidence"] = int(3)
                elif any(
                    i in used_keys
                    for i in [k for k, v in confidence_dict.items() if v == 2]
                ):
                    datapart.at[index, "confidence"] = int(2)
                else:
                    datapart.at[index, "confidence"] = int(1)
            else:
                # iterate over features to buffer the lines
                buffer_dist = None
                for key in used_keys:
                    combined_key = f"{key}={row[key][index]}"
                    if combined_key in buffer_dict:
                        buffer_dist = buffer_dict[combined_key]
                if buffer_dist is not None:
                    # buffer feature. Divide by 2, as the input defines the buffer radius
                    row["geometry"] = row.geometry.buffer(buffer_dist / 2)
                    # reproject feature back to WGS 84 to be able to add them to polygone features
                    row = row.to_crs(4326)
                    # add feature to df of buffered features
                    buffered_linefeatures = pandas.concat([buffered_linefeatures, row])

        if counter == 1:
            # if features are a line features, write confidence level 2
            datapart = buffered_linefeatures
            datapart["confidence"] = int(2)

        df_of_features = pandas.concat([df_of_features, datapart])
        counter += 1

    return df_of_features


def write_as_raster(df: gpd.GeoDataFrame, rastertile: pathlib.Path, filter_class: str, time: str) -> None:
    wc_data = rioxarray.open_rasterio(rastertile)
    try:
        osm_raster = make_geocube(vector_data=df, measurements=["confidence"], like=wc_data)
        tilename = get_tilename(rastertile.name)
        osm_raster_path = OSMRASTER + tilename + f"/{time}"
        if not os.path.exists(osm_raster_path):
            os.makedirs(osm_raster_path)
        osm_raster.rio.to_raster(osm_raster_path + f"/{filter_class}.tif")
    except geocube.exceptions.VectorDataError:
        print(f"No Data for Filter {filter_class} in Ratsertile {rastertile.name} in year {time}")

    wc_data = None


def query_osm_data(rastertile: str, extent: FeatureCollection, confidence_dict: dict, buffer_dict: dict, time: str) -> None:
    for filter_class in pathlib.Path(FILTERPATH).rglob("*.txt"):
        builtup_df = get_vector_areas(
            filter_class, time, extent, confidence_dict, buffer_dict
        )
        write_as_raster(builtup_df, rastertile, filter_class.stem, time[:4])

        # TODO: remove below. For Testing only
        #with open("./data/test/gdf.geojson", "w") as f:
        #    f.write(builtup_df.to_json())

        # TODO:
        #  dann muss raster draus gemacht werden, dass dem ursprünglichen entspricht
        #  Für andere Klassen wiederholen. Loop mit for every txt in filter dir?
        #  Für zweiten Zeitpunkt wiederholen, im Namen einbringen


def main():
    for file in pathlib.Path(TERRADIR + "Maps/").rglob("*_Map.tif"):
        print(f"started with {file.name}")

        bound_featurecol = get_extent(file)
        confidence_dict = load_dict(CONFICENCEDICT)
        buffer_dict = load_dict(BUFFERSIZES)
        #TODO: add for both years
        query_osm_data(file, bound_featurecol, confidence_dict, buffer_dict, TIME)
        # combine_rasters (per tile and time)

        print(f"finished with {file}")
        # TODO: remove below
        break

    # TODO:
    #  Zeit der Abfrage anpassen
    #  functionen weiter aufsplitten
    #  logging überarbeiten
    #  comments und docstrings adden


if __name__ == "__main__":
    main()
