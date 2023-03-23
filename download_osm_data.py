import os

import pandas
import rasterio
from geojson import FeatureCollection, Feature
import geojson
import geopandas as gpd
import requests
from shapely.geometry import box

from config import TERRADIR, APIENDPOINT, TIME, FILTERPATH


def get_tilename(file: str) -> str:
    name = file[-15:-8]
    return name


def get_extent(file: str) -> FeatureCollection:
    with rasterio.open(TERRADIR + "Maps/" + file) as raster:
        bounds = raster.bounds
    geom = box(*bounds)
    feature_collection = FeatureCollection([Feature(geometry=geom)])
    return feature_collection


def query_ohsome(path_to_filter: str, time: str, extent: FeatureCollection) -> gpd.GeoDataFrame:
    df_of_features = gpd.GeoDataFrame()
    with open(path_to_filter) as f:
        lines = f.readlines()

    confidence = 1
    for line in lines:
        osmfilter = line
        if len(osmfilter) != 0 and confidence != 4:
            filterquery = f"({osmfilter}) and geometry:polygon"
        elif len(osmfilter) != 0 and confidence == 4:
            #TODO: fix this
            # confidence muss runter. Auf 2?
            print("linefeature")
            filterquery = f"(lake=seilersee) and geometry:line"
        else:
            confidence += 1
            continue

        data = {"bpolys": geojson.dumps(extent), "time": time, "filter": filterquery, "properties": "tags"}
        response = requests.post(APIENDPOINT, data=data)
        response.raise_for_status()

        datapart = gpd.GeoDataFrame.from_features(response.json()['features'])
        datapart['confidence'] = confidence
        df_of_features = pandas.concat([df_of_features, datapart])

        confidence += 1

    return df_of_features


def query_builtup_data(tilename, extent):
    path_to_filter = FILTERPATH + "builtup.txt"
    builtup_df = query_ohsome(path_to_filter, TIME, extent)

    # TODO: remove below
    with open('./data/test/gdf.geojson', "w") as f:
        f.write(builtup_df.to_json())


    #properties = response.json()["properties"]
    """
    Zeit anpassen der abfrage
    jetzt habe ich alle, will das ja aber nicht, sondern in klassen. also muss ich filter schon aufteilen?
    line_feature fehlen
    dann muss raster draus gemacht werden, dass dem ursprünglichen entspricht
    """


def main():
    for file in os.listdir(TERRADIR + "Maps/"):
        print(f"started with {file}")
        tilename = get_tilename(file)
        bound_featurecol = get_extent(file)
        query_builtup_data(tilename, bound_featurecol)
        print(f"finished with {file}")
        break


if __name__ == "__main__":
    main()