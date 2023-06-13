import logging
import logging.config
import os
import pathlib

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
import yaml
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from rasterio import features
from shapely.geometry import box, shape
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

from config import (
    CM_PATH,
    COMP_PATH,
    INFILES,
    INPUTDIR,
    LOGGCONFIG,
    OSM_COMP_PATH,
    OSMRASTER,
    TERRADIR,
    WC_COMP_PATH,
    WC_OSM_COMP_PATH,
)
from download_worldcover_data import import_geodata


def get_tileyear(file: pathlib.Path) -> str:
    year = file.stem[-23:-19]
    return year


def get_tilename(file: pathlib.Path) -> str:
    name = file.stem[-13:-4]
    file_no = int(file.stem[-12:-9])
    feature = int(file.stem[-7:-4])  # TODO: check if actually used
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
    logger.warning(
        f"Could not find corresponding WorldCover Raster for WC File {raster}. Cannot compare WC Data"
    )
    return None


def get_other_to_compare(raster, datapath):
    for file in datapath.rglob("*.tif"):
        year = get_tileyear(raster)
        tile = get_tilename(raster)
        if year == get_osmyear(file) and tile == get_osmtile(file):
            return file
    logger.warning(
        f"Could not find corresponding OSM Raster for WC File {raster}. Cannot compare Data between OSM and WC"
    )
    return None


def get_osm_to_compare(raster, datapath):
    for file in datapath.rglob("*.tif"):
        year = get_osmyear(file)
        tile = get_osmtile(file)
        if year == "2020" and tile == get_osmtile(raster):
            return file
    logger.warning(
        f"Could not find corresponding OSM Raster for OSM File {raster}. Cannot compare OSM Data"
    )
    return None


def get_rasterdata(rasterpath, comparepath):
    with rasterio.open(rasterpath) as raster:
        rasterdata = raster.read()
    with rasterio.open(comparepath) as raster:
        comparedata = raster.read()

    return rasterdata, comparedata


def save_raster(data, path, crs, transform):
    new_dataset = rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        compress="lzw",
    )
    new_dataset.write(data)
    new_dataset.close()


def detect_equality(rasterpath, comparepath, resultfile):
    rasterdata, comparedata = get_rasterdata(rasterpath, comparepath)
    # binary_change = np.equal(rasterdata, comparedata)
    # use isclose with a tolerance of 0 as it can compare nan values (both nan -> no change)
    binary_change = np.isclose(rasterdata, comparedata, rtol=0, atol=0, equal_nan=True)
    with rasterio.open(rasterpath) as raster:
        save_raster(
            binary_change.astype(np.uint8), resultfile, raster.crs, raster.transform
        )
    logger.info(f"Wrote binary change raster {resultfile.name}")

    # calculate statistics
    no_of_pixel = binary_change.size
    pixel_matching = binary_change.sum()
    percentage_matching = pixel_matching / no_of_pixel * 100
    # percentage_deviation = np.invert(binary_change).sum()/no_of_pixel*100
    nan_pixel_compare = np.count_nonzero(comparedata == 999)
    completeness_percentage_compare = 100 - ((nan_pixel_compare / no_of_pixel) * 100)

    del rasterdata, comparedata, binary_change

    return (
        no_of_pixel,
        pixel_matching,
        percentage_matching,
        nan_pixel_compare,
        completeness_percentage_compare,
    )


def detect_loss_of_nature(rasterpath, comparepath, resultfile):
    """
    Get "old" classification where raster was not classified as built up but is now.
    :param rasterpath:
    :param comparepath:
    :param resultfile:
    :return: None
    """
    rasterdata, comparedata = get_rasterdata(rasterpath, comparepath)
    outputdata = np.where((rasterdata == 50) & (comparedata != 50), comparedata, 0)
    with rasterio.open(rasterpath) as raster:
        save_raster(outputdata, resultfile, raster.crs, raster.transform)
    logger.info(
        f"Wrote raster indicating change of class to built up called {resultfile.name}"
    )

    loss_of_nature_vector(rasterpath, outputdata, resultfile)

    del rasterdata, comparedata, outputdata


def loss_of_nature_vector(sourcepath, data, resultfile):
    with rasterio.open(sourcepath) as raster:
        transform = raster.transform
    mask = data != 0
    shapes = features.shapes(data.astype(np.uint16), mask=mask, transform=transform)

    classcode = []
    geometry = []
    for shapedict, value in shapes:
        classcode.append(value)
        geometry.append(shape(shapedict))

    # build the gdf object over the two lists
    gdf = gpd.GeoDataFrame(
        {"old_class": classcode, "geometry": geometry}, crs="EPSG:4326"
    )
    gdf["new_class"] = int(50)

    # reproject to utm to calculate area
    # get example coordinate to query useable UTM projection
    example_coords = gdf["geometry"][0].bounds
    # query UTM code for corner coordinate
    utm_crs_list = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(
            west_lon_degree=example_coords[0],
            south_lat_degree=example_coords[1],
            east_lon_degree=example_coords[0],
            north_lat_degree=example_coords[1],
        ),
    )
    utm_code = utm_crs_list[0].code
    # reproject feature to queried UTM
    gdf.to_crs(utm_code, inplace=True)

    gdf["area"] = gdf.area

    # reproject back to wgs 84
    gdf.to_crs(4326, inplace=True)

    outputpath = pathlib.Path(f"{resultfile.parent}/{resultfile.stem}.geojson")

    with open(outputpath, "w") as f:
        f.write(gdf.to_json())

    logger.info(
        f"Wrote vector indicating change of class to built up called {outputpath.name}"
    )


def create_cm(rasterdata, comparedata, year):
    # calculate confusion matrix. First Position: actual, Second: predicted
    actual = np.nan_to_num(rasterdata.flatten(), nan=999)
    pred = np.nan_to_num(comparedata.flatten(), nan=999)

    # create confusion Matrix using No. of Pixel
    df_confusion_pandas = pd.crosstab(
        actual, pred, rownames=["WC Classes"], colnames=["OSM Classes"], margins=True
    )
    df_confusion_pandas = df_confusion_pandas.reindex(
        index=[10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100, "All"],
        columns=[10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100, 999, "All"],
        fill_value=0,
    )

    fig, ax = plt.subplots(figsize=(20, 15))
    cm_map = sns.heatmap(
        df_confusion_pandas,
        annot=True,
        fmt="",
        robust=True,
        annot_kws={"fontsize": 9},
        square=True,
        ax=ax,
        cmap="Blues",
    )
    ax.set_xlabel("OSM Classes", labelpad=10)
    ax.set_ylabel("WC Classes", labelpad=10)
    plt.title(f"Confusion Matrix stating No. of Pixel (WC vs. OSM {year})")
    border_linewidth = 1
    for _, spine in cm_map.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(border_linewidth)
    # plt.show()
    # plt.close()
    if not os.path.exists(CM_PATH):
        os.makedirs(CM_PATH)
    save_path_norm = pathlib.Path(f"{CM_PATH}/CM_WCvsOSM_{year}_absolut.png")
    plt.savefig(save_path_norm)

    # calculate CM with normalized Values
    fig, ax = plt.subplots(figsize=(20, 15))
    cm_display = ConfusionMatrixDisplay.from_predictions(
        actual,
        pred,
        labels=[10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100],
        normalize="true",
        values_format=".3f",
        cmap="Blues",
        ax=ax,
    )
    cm_display.ax_.set(xlabel="OSM Classes", ylabel="WC Classes")
    plt.title(f"Confusion Matrix stating relative Values (WC vs. OSM {year})")
    save_path_norm = pathlib.Path(f"{CM_PATH}/CM_WCvsOSM_{year}_normalised.png")
    plt.savefig(save_path_norm)
    logger.info(f"Wrote Confusion Matrices for year {year}")

    cm_report = classification_report(actual, pred, zero_division=0)
    save_path_report = pathlib.Path(f"{CM_PATH}/CM_WCvsOSM_{year}_Report.txt")
    with open(save_path_report, "w") as file:
        file.write(cm_report)


def main(compare_wc=True, compare_wc_osm=True, compare_osm=True):
    wc_datapath = pathlib.Path(TERRADIR + "Maps/")
    osm_datapath = pathlib.Path(OSMRASTER)

    if len(INFILES) == 0:
        for filename in os.listdir(INPUTDIR):
            if filename.endswith(".geojson"):
                INFILES.append(filename)

    # create a vector which holds statistics for all raster
    statistics = gpd.GeoDataFrame()

    file_counter = 0
    for infile in INFILES:
        logger.info(f"Working on Inputfile {file_counter + 1} of {len(INFILES)}")
        # import geodata of extent to be imported from WorldCover
        gdf = import_geodata(INPUTDIR, infile)

        for index in gdf.index:
            logger.info(f"Working on Feature {index+1} of {len(gdf)}")

            # Take WC Raster to get extent from to create vector file and save statistics to it
            rastername_finder = (
                f"ESA_WorldCover_10m_2021_v100_f{file_counter:03}id{index:03}_Map.tif"
            )
            rasterpath = pathlib.Path(wc_datapath / rastername_finder)
            with rasterio.open(rasterpath) as raster:
                bounds = raster.bounds
            box_geom = box(*bounds)
            # create a vector feature for this AoI to hold all stats and be merged into the general one
            feat_stats = gpd.GeoDataFrame(
                {"file_no": file_counter, "feature_no": index, "geometry": box_geom},
                crs="EPSG:4326",
                index=[0],
            )

            if compare_wc:
                rastername_finder = f"ESA_WorldCover_10m_2021_v100_f{file_counter:03}id{index:03}_Map.tif"
                rasterpath = pathlib.Path(wc_datapath / rastername_finder)
                comparepath = get_wc_to_compare(rasterpath, wc_datapath)
                if comparepath is not None:
                    resultdir = WC_COMP_PATH
                    if not os.path.exists(resultdir):
                        os.makedirs(resultdir)
                    # binary change
                    resultfile = pathlib.Path(
                        resultdir + f"{get_tilename(rasterpath)}_wc_binary_change.tif"
                    )
                    (
                        feat_stats["pixel_count"],
                        feat_stats["WC_Match_Pixel"],
                        feat_stats["WC_Match_Percent"],
                    ) = detect_equality(rasterpath, comparepath, resultfile)[0:3]

                    # old classes have become built up
                    resultfile = pathlib.Path(
                        resultdir + f"{get_tilename(rasterpath)}_wc_loss_of_nature.tif"
                    )
                    detect_loss_of_nature(rasterpath, comparepath, resultfile)
                else:
                    logger.error(
                        f"Cannot find WC Raster to be compared with {rasterpath}"
                    )
            if compare_wc_osm:
                for rasterpath in wc_datapath.rglob(
                    f"*f{file_counter:03}id{index:03}_Map.tif"
                ):
                    comparepath = get_other_to_compare(rasterpath, osm_datapath)
                    if comparepath is not None:
                        resultdir = WC_OSM_COMP_PATH
                        if not os.path.exists(resultdir):
                            os.makedirs(resultdir)
                        tile_year = get_tileyear(rasterpath)
                        resultfile = pathlib.Path(
                            resultdir
                            + f"{get_tilename(rasterpath)}_{tile_year}_wc_osm_binary_change.tif"
                        )
                        (
                            feat_stats[f"osm_acc_{tile_year}"],
                            feat_stats[f"osm_nan_pixel_{tile_year}"],
                            feat_stats[f"osm_completeness_{tile_year}"],
                        ) = detect_equality(rasterpath, comparepath, resultfile)[2:]

                        # create confusion matrices
                        wc_data, osm_data = get_rasterdata(rasterpath, comparepath)
                        create_cm(wc_data, osm_data, get_tileyear(rasterpath))
                        del wc_data, osm_data
                    else:
                        logger.error(
                            f"Cannot find OSM Raster to be compared with WC Raster {rasterpath}"
                        )
            if compare_osm:
                rastername_finder = f"f{file_counter:03}id{index:03}_2021.tif"
                rasterpath = pathlib.Path(osm_datapath / rastername_finder)
                comparepath = get_osm_to_compare(rasterpath, osm_datapath)
                if comparepath is not None:
                    resultdir = OSM_COMP_PATH
                    if not os.path.exists(resultdir):
                        os.makedirs(resultdir)
                    # binary change
                    resultfile = pathlib.Path(
                        resultdir + f"{get_osmtile(rasterpath)}_osm_binary_change.tif"
                    )
                    (
                        feat_stats["pixel_count"],
                        feat_stats["OSM_Match_Pixel"],
                        feat_stats["OSM_Match_Percent"],
                    ) = detect_equality(rasterpath, comparepath, resultfile)[0:3]
                    # old classes have become built up
                    resultfile = pathlib.Path(
                        resultdir + f"{get_osmtile(rasterpath)}_osm_loss_of_nature.tif"
                    )
                    detect_loss_of_nature(rasterpath, comparepath, resultfile)
                else:
                    logger.error(
                        f"Cannot find OSM Raster to be compared with {rasterpath}"
                    )

            statistics = pd.concat([statistics, feat_stats], ignore_index=True)

    stats_outpath = pathlib.Path(f"{COMP_PATH}/statistics.geojson")
    with open(stats_outpath, "w") as file:
        file.write(statistics.to_json())


if __name__ == "__main__":
    # warnings.filterwarnings("error")
    with open(LOGGCONFIG, "r") as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    logger = logging.getLogger(__name__)

    main(compare_wc=True, compare_wc_osm=True, compare_osm=True)

    # TODO:
    #  add logging
    #  add typehint
