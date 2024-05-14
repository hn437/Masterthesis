import logging
import logging.config
import pathlib


import geopandas as gpd
import requests


import yaml

from config import COMP_PATH, LOGGCONFIG, URL, HEADERS
from download_worldcover_data import import_geodata


def main() -> None:
    """
    Calculates Mapping Saturation for each AoI feature in the statistics.geojson file.
    :return: None
    """

    # Read the statistics file which contains all AoI features and their Statistics
    infile = "statistics.geojson"
    gdf = import_geodata(COMP_PATH, infile)

    # Iterate over each AoI feature to calculate Indicator for each Filter
    for index in gdf.index:
        logging.info(f"Working on Feature No {index}")
        # Extract the geometry from the selected row
        geometry = gdf.loc[index, "geometry"]

        # Create a GeoSeries from the geometry
        geo_series = gpd.GeoSeries(geometry)

        # Convert the GeoSeries to GeoJSON format
        geojson_feature = geo_series.__geo_interface__

        # Create a GeoJSON feature collection as required by the API
        bpolys = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": geojson_feature["features"][0]["geometry"],
                }
            ],
        }

        # Iterate over each filter to query the Indicator from the API
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

            # Query Indicator for the selected filter and area
            response = requests.post(URL, headers=HEADERS, json=parameters)
            # Raise an Exception if HTTP Status Code is not 200
            response.raise_for_status()
            result = response.json()["result"][0]["result"]

            # Save the Indicator value & class per filter in the Statistics GeoDataFrame
            gdf.at[index, f"{filter}_value"] = result["value"]
            gdf.at[index, f"{filter}_class"] = result["class"]

    logging.info("Done calculating Mapping Saturations!")

    # Save the updated statistics GeoDataFrame as a GeoJSON file
    outpath = pathlib.Path(f"{COMP_PATH}/statistics.geojson")
    with open(outpath, "w") as file:
        file.write(gdf.to_json())
    logging.info(f"File saved as {outpath}.")

    # Save the updated statistics GeoDataFrame as an Excel file
    outpath_excel = pathlib.Path(f"{COMP_PATH}/statistics.xlsx")
    gdf.to_excel(outpath_excel, index=False)
    logging.info(f"Excel File saved as {outpath_excel}. Done!")


if __name__ == "__main__":
    # warnings.filterwarnings("error")
    with open(LOGGCONFIG, "r") as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)

    main()
