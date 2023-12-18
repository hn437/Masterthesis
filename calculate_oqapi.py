import logging
import logging.config
import pathlib


import geopandas as gpd
import requests


import yaml

from config import COMP_PATH, LOGGCONFIG, URL, HEADERS
from download_worldcover_data import import_geodata


def main():
    infile = "statistics.geojson"
    gdf = import_geodata(COMP_PATH, infile)

    for index in gdf.index:
        logging.info(f"Working on Feature No {index}")
        # Extract the geometry from the selected row
        geometry = gdf.loc[index, "geometry"]

        # Create a GeoSeries from the geometry
        geo_series = gpd.GeoSeries(geometry)

        # Convert the GeoSeries to GeoJSON format
        geojson_feature = geo_series.__geo_interface__

        # Create a GeoJSON feature collection
        bpolys = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": geojson_feature["features"][0]["geometry"],
                }
            ],
        }

        for filter in [
            "all-filter",
            "bare",
            "built-up",
            "cropland",
            "grassland",
            "mangroves",
            "moss",
            "shrubland",
            "snow",
            "tree",
            "water",
            "wetland",
            "garden",
        ]:
            logging.info(f"Calculating Mapping Saturation for Filter {filter}")

            parameters = {
                "topic": filter,
                "bpolys": bpolys,
            }

            response = requests.post(URL, headers=HEADERS, json=parameters)
            response.raise_for_status()  # Raise an Exception if HTTP Status Code is not 200
            result = response.json()["result"][0]["result"]

            gdf.at[index, f"{filter}_value"] = result["value"]
            gdf.at[index, f"{filter}_class"] = result["class"]

    logging.info("Done calculating Mapping Saturations!")
    outpath = pathlib.Path(f"{COMP_PATH}/statistics.geojson")
    with open(outpath, "w") as file:
        file.write(gdf.to_json())
    logging.info(f"File saved as {outpath}.")
    outpath_excel = pathlib.Path(f"{COMP_PATH}/statistics.xlsx")
    gdf.to_excel(outpath_excel, index=False)
    logging.info(f"Excel File saved as {outpath_excel}. Done!")


if __name__ == "__main__":
    # warnings.filterwarnings("error")
    with open(LOGGCONFIG, "r") as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)

    main()
