import os

import pandas
import rasterio
from geojson import FeatureCollection, Feature
import geojson
import geopandas as gpd
import requests
from shapely.geometry import box

from config import TERRADIR, APIENDPOINT, TIME, FILTERPATH, CONFICENCEDICT, BUFFERSIZES


def get_tilename(file: str) -> str:
    name = file[-15:-8]
    return name


def get_extent(file: str) -> FeatureCollection:
    with rasterio.open(TERRADIR + "Maps/" + file) as raster:
        bounds = raster.bounds
    geom = box(*bounds)
    feature_collection = FeatureCollection([Feature(geometry=geom)])
    return feature_collection


def load_dict(path) -> dict:
    d = {}
    with open(path) as f:
        for line in f:
            (key, val) = line.split(",")
            d[key] = int(val)

    return d


def query_ohsome(path_to_filter: str, time: str, extent: FeatureCollection, confidence_dict: dict, buffer_dict: dict) -> gpd.GeoDataFrame:
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

        data = {"bpolys": geojson.dumps(extent), "time": time, "filter": filterquery, "properties": "tags"}
        response = requests.post(APIENDPOINT, data=data)
        response.raise_for_status()

        datapart = gpd.GeoDataFrame.from_features(response.json()['features'])

        # dropping all columns (= OSM Keys) not queried
        column_names = datapart.columns.values.tolist()[3:]
        confidence_keys = [key for key in confidence_dict]
        col_to_drop = list(set(column_names) - set(confidence_keys))
        datapart = datapart.drop(columns=col_to_drop)

        # iterate over features to assign confidence level and buffer lines
        for index in datapart.index:
            row = datapart.loc[[index]]
            row = row[row.columns[~row.isnull().all()]]
            used_keys  = row.columns.values.tolist()[3:]

            if counter == 0:
            # iterate over features to assign confidence level of polygons
                if any(i in used_keys for i in [k for k, v in confidence_dict.items() if v == 3]):
                    datapart.at[index, "confidence"] = int(3)
                elif any(i in used_keys for i in [k for k, v in confidence_dict.items() if v == 2]):
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
                    row['geometry'] = row.geometry.buffer(buffer_dist)
                    buffered_linefeatures = pandas.concat([buffered_linefeatures, row])

        if counter ==1:
            datapart = buffered_linefeatures
            datapart["confidence"] = int(2)

        df_of_features = pandas.concat([df_of_features, datapart])
        counter += 1

    return df_of_features


def query_builtup_data(tilename, extent, confidence_dict, buffer_dict):
    path_to_filter = FILTERPATH + "builtup.txt"
    builtup_df = query_ohsome(path_to_filter, TIME, extent, confidence_dict, buffer_dict)

    # TODO: remove below
    with open('./data/test/gdf.geojson', "w") as f:
        f.write(builtup_df.to_json())


    #properties = response.json()["properties"]
    """
    Zeit anpassen der abfrage
    jetzt habe ich alle, will das ja aber nicht, sondern in klassen. also muss ich filter schon aufteilen?
    line_feature müssen gebuffert werden
    dann muss raster draus gemacht werden, dass dem ursprünglichen entspricht
    """


def main():
    for file in os.listdir(TERRADIR + "Maps/"):
        print(f"started with {file}")

        tilename = get_tilename(file)
        bound_featurecol = get_extent(file)
        confidence_dict = load_dict(CONFICENCEDICT)
        buffer_dict = load_dict(BUFFERSIZES)
        query_builtup_data(tilename, bound_featurecol, confidence_dict, buffer_dict)

        print(f"finished with {file}")
        break


if __name__ == "__main__":
    main()