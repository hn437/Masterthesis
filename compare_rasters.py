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
    """
    Determines the year which is represented by a WC rasterfile by its name
    :param file: path to rasterfile
    :return: string of the year
    """
    year = file.stem[-23:-19]
    return year


def get_tilename(file: pathlib.Path) -> str:
    """
    Determines the tile which is represented by a WC rasterfile by its name
    :param file: path to rasterfile
    :return: string of tile identifier
    """
    name = file.stem[-13:-4]
    int(file.stem[-12:-9])
    int(file.stem[-7:-4])
    return name


def get_osmyear(file: pathlib.Path) -> str:
    """
    Determines the year which is represented by an OSM rasterfile by its name
    :param file: path to rasterfile
    :return: string of the year
    """
    year = file.stem[-4:]
    return year


def get_osmtile(file: pathlib.Path) -> str:
    """
    Determines the tile which is represented by an OSM rasterfile by its name
    :param file: path to rasterfile
    :return: string of tile identifier
    """
    name = file.stem[:-5]
    return name


def get_wc_to_compare(raster: pathlib.Path, datapath: pathlib.Path) -> pathlib.Path or None:
    """
    Function which determines the respective WC raster for the year 2020 for a given WC
    raster of the year 2021
    :param raster: raster for which the respective WC raster of the year 2020 should be
    found
    :param datapath: path to dir of WC rasters
    :return: path to respective WC raster of the year 2020 or None if no corresponding
    raster exists
    """
    # iterate over all files in the directory and find the corresponding raster by
    # matching tile identifier and year required
    for file in datapath.rglob("*_Map.tif"):
        year = get_tileyear(file)
        tile = get_tilename(file)
        if year == "2020" and tile == get_tilename(raster):
            return file
    logging.warning(
        f"Could not find corresponding WorldCover Raster for WC File {raster}. Cannot"
        f"compare WC Data"
    )
    return None


def get_other_to_compare(raster: pathlib.Path, datapath: pathlib.Path) -> pathlib.Path or None:
    """
    Function which determines the respective OSM raster for the same year for a given WC
    raster
    :param raster: the WC raster for which the respective OSM raster should be found
    :param datapath: path to dir of OSM rasters
    :return: path to respective OSM raster of the same year or None if no corresponding
    raster exists
    """
    # iterate over all files in the directory and find the corresponding raster by
    # matching tile identifier and matching year
    for file in datapath.rglob("*.tif"):
        year = get_tileyear(raster)
        tile = get_tilename(raster)
        if year == get_osmyear(file) and tile == get_osmtile(file):
            return file
    logging.warning(
        f"Could not find corresponding OSM Raster for WC File {raster}. Cannot compare"
        f"Data between OSM and WC"
    )
    return None


def get_osm_to_compare(raster: pathlib.Path, datapath: pathlib.Path)  -> pathlib.Path or None:
    """
    Function which determines the respective OSM raster for the year 2020 for a given
    OSM raster of the year 2021
    :param raster: raster for which the respective OSM raster of the year 2020 should be
    found
    :param datapath: path to dir of OSM rasters
    :return: path to respective OSM raster of the year 2020 or None if no corresponding
    raster exists
    """
    # iterate over all files in the directory and find the corresponding raster by
    # matching tile identifier and year required
    for file in datapath.rglob("*.tif"):
        year = get_osmyear(file)
        tile = get_osmtile(file)
        if year == "2020" and tile == get_osmtile(raster):
            return file
    logging.warning(
        f"Could not find corresponding OSM Raster for OSM File {raster}. Cannot compare"
        f"OSM Data"
    )
    return None


def get_rasterdata(rasterpath: pathlib.Path, comparepath: pathlib.Path) -> tuple[np.array, np.array]:
    """
    Function to read raster data from two rasterfiles which should be compared
    :param rasterpath: raster stating the new data
    :param comparepath: base raster to which to compare against
    :return: tuple of two numpy arrays containing the raster data
    """
    with rasterio.open(rasterpath) as raster:
        rasterdata = raster.read()
    with rasterio.open(comparepath) as raster:
        comparedata = raster.read()

    return rasterdata, comparedata


def save_raster(data: np.array, path: pathlib.Path, crs, transform) -> None:
    """
    Function to save data of an array as a rasterfile
    :param data: array of the data to be written as
    :param path: path to the file to be written
    :param crs: coordinate reference system of the raster
    :param transform: affine transformation of the raster
    :return: None
    """
    # create new raster file
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
    # write data into new file
    new_dataset.write(data)
    new_dataset.close()


def detect_equality(rasterpath: pathlib.Path, comparepath: pathlib.Path, resultfile: pathlib.Path) -> tuple[int, int, float, int, float, int, float, float, float]:
    """
    Function to detect equality of two rasterfiles a.k.a accuracy as well as determines
    completeness of the files
    :param rasterpath: path to the main raster
    :param comparepath: path to the raster to be compared against
    :param resultfile: path were new file will be created to indicate matching pixels
    :return: tuple of all calculated statistics
    """
    # read in data of the rasters to be compared
    rasterdata, comparedata = get_rasterdata(rasterpath, comparepath)

    # write 1 where rasterdata is equal to comparedata, 0 where not
    # use np.isclose with a tolerance of 0 as it can compare nan values (both nan -> no
    # change)
    binary_change = np.isclose(rasterdata, comparedata, rtol=0, atol=0, equal_nan=True)
    # write binary change raster to drive
    with rasterio.open(rasterpath) as raster:
        save_raster(
            binary_change.astype(np.uint8), resultfile, raster.crs, raster.transform
        )
    logging.info(f"Wrote binary change raster {resultfile.name}")

    # aggregate vegetation classes
    rasterdata = np.where(
        (rasterdata == 10)
        | (rasterdata == 20)
        | (rasterdata == 30)
        | (rasterdata == 40)
        | (rasterdata == 90)
        | (rasterdata == 95)
        | (rasterdata == 100),
        120,
        rasterdata,
    )
    comparedata = np.where(
        (comparedata == 10)
        | (comparedata == 20)
        | (comparedata == 30)
        | (comparedata == 40)
        | (comparedata == 90)
        | (comparedata == 95)
        | (comparedata == 100),
        120,
        comparedata,
    )
    # write 1 where rasterdata is equal to comparedata, 0 where not
    binary_change_aggregated = np.isclose(
        rasterdata, comparedata, rtol=0, atol=0, equal_nan=True
    )
    # write binary change raster of aggregated classes comparison to drive
    resultfile_aggregated = pathlib.Path(
        f"{resultfile.parent}/{resultfile.stem}_aggregated_classes.tif"
    )
    with rasterio.open(rasterpath) as raster:
        save_raster(
            binary_change_aggregated.astype(np.uint8),
            resultfile_aggregated,
            raster.crs,
            raster.transform,
        )
    logging.info(f"Wrote binary change raster {resultfile_aggregated.name}")

    # calculate statistics

    # total number of  pixels
    no_of_pixel = binary_change.size
    # number of matching pixels
    pixel_matching = binary_change.sum()
    # number of matching pixels for aggregated classes
    pixel_matching_aggregated = binary_change_aggregated.sum()
    # percentage of matching pixels
    percentage_matching = pixel_matching / no_of_pixel * 100
    # percentage of matching pixels for aggregated classes
    percentage_matching_aggregated = pixel_matching_aggregated / no_of_pixel * 100
    # number of nan pixels in comparedata
    nan_pixel_compare = np.count_nonzero(comparedata == 999)
    # percentage of completeness of comparedata
    completeness_percentage_compare = 100 - ((nan_pixel_compare / no_of_pixel) * 100)
    # percentage of matching pixels without nan values
    percentage_matching_no_nan = (
        pixel_matching / (no_of_pixel - nan_pixel_compare) * 100
    )
    # percentage of matching pixels for aggregated classes without nan values
    percentage_matching_aggregated_no_nan = (
        pixel_matching_aggregated / (no_of_pixel - nan_pixel_compare) * 100
    )

    del rasterdata, comparedata, binary_change

    return (
        no_of_pixel,
        pixel_matching,
        percentage_matching,
        pixel_matching_aggregated,
        percentage_matching_aggregated,
        nan_pixel_compare,
        completeness_percentage_compare,
        percentage_matching_no_nan,
        percentage_matching_aggregated_no_nan,
    )


def detect_loss_of_nature(rasterpath: pathlib.Path, comparepath: pathlib.Path, resultfile: pathlib.Path) -> np.array:
    """
    This functions detects pixels which changed from another LULC class or noData to
    built-up in a dataset
    :param rasterpath: raster of the newer year
    :param comparepath: raster of the older year
    :param resultfile: path for the new rasterfile to be created indicating change to
    built-up
    :return: an array with zeros where no change to built-up happened and the original
    LULC class otherwise
    """
    # read in data of the rasters to be compared
    rasterdata, comparedata = get_rasterdata(rasterpath, comparepath)
    # write 0 where class built-up was already present or is not present in newer dataset
    # write one where loss of nature was detected
    outputdata = np.where((rasterdata == 50) & (comparedata != 50), 1, 0)

    # reassign the original classes where loss of nature happened
    outputdata = np.where((outputdata == 1), comparedata, 0)

    # write raster indicating change to built-up to drive
    with rasterio.open(rasterpath) as raster:
        save_raster(outputdata, resultfile, raster.crs, raster.transform)
    logging.info(
        f"Wrote raster indicating change of class to built up called {resultfile.name}"
    )
    # write areas which changed from another class to built-up as vector
    loss_of_nature_vector(rasterpath, outputdata, resultfile)
    # aggregated vegetation classes
    aggregated_data = np.where(
        (outputdata == 10)
        | (outputdata == 20)
        | (outputdata == 30)
        | (outputdata == 40)
        | (outputdata == 90)
        | (outputdata == 95)
        | (outputdata == 100),
        120,
        outputdata,
    )
    # write raster indicating change to built-up using aggregated classes to drive
    aggregated_outpath = pathlib.Path(
        f"{resultfile.parent}/{resultfile.stem}_aggregated_vegetation.tif"
    )
    with rasterio.open(rasterpath) as raster:
        save_raster(aggregated_data, aggregated_outpath, raster.crs, raster.transform)
    logging.info(
        "Wrote raster indicating change of class to built up with aggregated nature class"
    )

    del rasterdata, comparedata, outputdata

    return aggregated_data


def loss_of_nature_vector(sourcepath: pathlib.Path, data: np.array, resultfile: pathlib.Path) -> None:
    """
    Function to write areas where change to the class built-up happened as vector data
    :param sourcepath: path to the original rasterfile
    :param data: array with zeroes where no such change occured and otherwise the
    original class code
    :param resultfile: path to the file to be written
    :return: None
    """
    # get affine transformation of the original raster
    with rasterio.open(sourcepath) as raster:
        transform = raster.transform
    # make a mask showing all areas where such change happened
    mask = data != 0
    # get the shapes where change happened and the value the data shows at that place
    shapes = features.shapes(data.astype(np.uint16), mask=mask, transform=transform)

    # create two lists for the class codes and the geometries for each shape
    classcode = []
    geometry = []
    for shapedict, value in shapes:
        classcode.append(value)
        geometry.append(shape(shapedict))

    # make a gdf from the geometries and values
    gdf = gpd.GeoDataFrame(
        {"old_class": classcode, "geometry": geometry}, crs="EPSG:4326"
    )
    # combine vegetation classes
    if len(gdf) > 0:
        gdf["agg_vegetation"] = np.where(
            (gdf["old_class"] == 10)
            | (gdf["old_class"] == 20)
            | (gdf["old_class"] == 30)
            | (gdf["old_class"] == 40)
            | (gdf["old_class"] == 90)
            | (gdf["old_class"] == 95)
            | (gdf["old_class"] == 100),
            120,
            gdf["old_class"],
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

        # write gdf as geojson file to drive
        outputpath = pathlib.Path(f"{resultfile.parent}/{resultfile.stem}.geojson")

        if not os.path.exists(resultfile.parent):
            os.makedirs(resultfile.parent)

        with open(outputpath, "w") as f:
            f.write(gdf.to_json())

        logging.info(
            f"Wrote vector indicating change of class to built up called {outputpath.name}"
        )
    else:
        logging.info(
            "No change of class to built up could be found so vectorfile could not be created."
        )


def create_cm(
    rasterdata, comparedata, tilename, year=None, aggregated=False, change_cm=False
):
    # calculate confusion matrix. First Position: actual, Second: predicted
    actual_raw = np.nan_to_num(rasterdata.flatten(), nan=999)
    del rasterdata
    pred_raw = np.nan_to_num(comparedata.flatten(), nan=999)
    del comparedata

    # only use data where OSM data is available
    actual = actual_raw[pred_raw != 999]
    pred = pred_raw[pred_raw != 999]

    if aggregated:
        actual = np.where(
            (actual == 10)
            | (actual == 20)
            | (actual == 30)
            | (actual == 40)
            | (actual == 90)
            | (actual == 95)
            | (actual == 100),
            120,
            actual,
        )
        pred = np.where(
            (pred == 10)
            | (pred == 20)
            | (pred == 30)
            | (pred == 40)
            | (pred == 90)
            | (pred == 95)
            | (pred == 100),
            120,
            pred,
        )

    # create confusion Matrix using No. of Pixel
    df_confusion_pandas = pd.crosstab(
        actual, pred, rownames=["WC Classes"], colnames=["OSM Classes"], margins=True
    )
    if aggregated:
        index_WC = [120, 50, 60, 70, 80, "All"]
        columns_OSM = [120, 50, 60, 70, 80, "All"]
    elif change_cm:
        index_WC = [0, 120, 50, 60, 70, 80, "All"]
        columns_OSM = [0, 120, 50, 60, 70, 80, "All"]
    else:
        index_WC = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100, "All"]
        columns_OSM = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100, "All"]

    df_confusion_pandas = df_confusion_pandas.reindex(
        index=index_WC,
        columns=columns_OSM,
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
    border_linewidth = 1
    for _, spine in cm_map.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(border_linewidth)
    if not os.path.exists(CM_PATH):
        os.makedirs(CM_PATH)
    if aggregated:
        plt.title(
            f"Confusion Matrix stating No. of Pixel for aggregated Classes (WC vs. OSM {year}) ({tilename})"
        )
        save_path_norm = pathlib.Path(
            f"{CM_PATH}/{tilename}_CM_WCvsOSM_aggregated_{year}_absolut.png"
        )
    elif change_cm:
        plt.title(
            f"Confusion Matrix stating No. of Pixel of Classes changed to Built-Up ({tilename})"
        )
        save_path_norm = pathlib.Path(f"{CM_PATH}/{tilename}_Class_Change_absolut.png")
    else:
        plt.title(
            f"Confusion Matrix stating No. of Pixel (WC vs. OSM {year}) ({tilename})"
        )
        save_path_norm = pathlib.Path(
            f"{CM_PATH}/{tilename}_CM_WCvsOSM_{year}_absolut.png"
        )
    plt.savefig(save_path_norm)
    plt.close()
    del df_confusion_pandas

    # calculate CM with normalized Values
    fig, ax = plt.subplots(figsize=(20, 15))
    cm_display = ConfusionMatrixDisplay.from_predictions(
        actual,
        pred,
        labels=columns_OSM[:-1],
        normalize="true",
        values_format=".4f",
        cmap="Blues",
        ax=ax,
    )
    cm_display.ax_.set(xlabel="OSM Classes", ylabel="WC Classes")
    if aggregated:
        plt.title(
            f"Confusion Matrix stating relative Values for aggregated Classes (WC vs. OSM {year}) ({tilename})"
        )
        save_path_norm = pathlib.Path(
            f"{CM_PATH}/{tilename}_CM_WCvsOSM_aggregated_{year}_normalised.png"
        )
    elif change_cm:
        plt.title(
            f"Confusion Matrix stating relative Values of Classes changed to Built-Up ({tilename})"
        )
        save_path_norm = pathlib.Path(
            f"{CM_PATH}/{tilename}_Class_Change_normalised.png"
        )
    else:
        plt.title(
            f"Confusion Matrix stating relative Values (WC vs. OSM {year}) ({tilename})"
        )
        save_path_norm = pathlib.Path(
            f"{CM_PATH}/{tilename}_CM_WCvsOSM_{year}_normalised.png"
        )
    plt.savefig(save_path_norm)
    plt.close()

    if change_cm:
        logging.info("Wrote Confusion Matrices for Class Change to Built-up")
    else:
        logging.info(
            f"Wrote Confusion Matrices for year {year}, aggregation={aggregated}"
        )

    cm_report = classification_report(actual, pred, zero_division=0, output_dict=False)
    cm_report_dict = classification_report(
        actual, pred, zero_division=0, output_dict=True
    )
    if aggregated:
        save_path_report = pathlib.Path(
            f"{CM_PATH}/{tilename}_CM_WCvsOSM_aggregated_{year}_Report.txt"
        )
    elif change_cm:
        save_path_report = pathlib.Path(f"{CM_PATH}/{tilename}_Class_Change_Report.txt")
    else:
        save_path_report = pathlib.Path(
            f"{CM_PATH}/{tilename}_CM_WCvsOSM_{year}_Report.txt"
        )
    with open(save_path_report, "w") as file:
        file.write(cm_report)

    # return report for built-up class
    return cm_report_dict.get("50")


def compare_change_area(rasterpath_wc, comparepath_wc, rasterpath_osm, comparepath_osm):
    tilename = get_tilename(rasterpath_wc)
    rasterdata_wc, comparedata_wc = get_rasterdata(rasterpath_wc, comparepath_wc)
    rasterdata_osm, comparedata_osm = get_rasterdata(rasterpath_osm, comparepath_osm)
    del rasterpath_wc, comparepath_wc, rasterpath_osm, comparepath_osm

    # write 0 where class built-up was already present or is not present in newer dataset
    # write 1 where change to built-up happened
    changedata_wc = np.where((rasterdata_wc == 50) & (comparedata_wc != 50), 1, 0)

    # comment out majority filter
    # changedata_wc = majority(changedata_wc, cube(MAJORITY_SIZE))

    del rasterdata_wc, comparedata_wc
    changedata_osm = np.where((rasterdata_osm == 50) & (comparedata_osm != 50), 1, 0)

    # also get a file which has Yes wherever WC changed to built up and OSM is built up
    #  in newer file
    wc_changed_built = np.where(
        (changedata_wc == 1) & (rasterdata_osm == 50), "Yes", "No"
    )
    del rasterdata_osm, comparedata_osm
    # write No where class built-up was already present or is not present in newer dataset
    # write Yes where change to built-up happened
    changedata_wc = np.where(changedata_wc == 1, "Yes", "No")
    changedata_osm = np.where(changedata_osm == 1, "Yes", "No")

    changedata_wc_masked = np.ma.masked_where(
        np.logical_and(changedata_wc == "No", changedata_osm == "No"), changedata_wc
    )
    changedata_osm_masked = np.ma.masked_where(
        np.logical_and(changedata_wc == "No", changedata_osm == "No"), changedata_osm
    )
    # only mask to where change in WC happened, as we only check whether the newer OSM
    #  data is built-up, but not if actual change happened.
    wc_changed_built_masked = np.ma.masked_where(
        changedata_wc == "No", wc_changed_built
    )
    del changedata_wc, changedata_osm, wc_changed_built

    actual = np.nan_to_num(changedata_wc_masked.flatten(), nan=999)
    pred = np.nan_to_num(changedata_osm_masked.flatten(), nan=999)
    pred_2 = np.nan_to_num(wc_changed_built_masked.flatten(), nan=999)
    del changedata_wc_masked, changedata_osm_masked, wc_changed_built_masked

    # create confusion Matrix using No. of Pixel
    df_confusion_pandas = pd.crosstab(
        actual, pred, rownames=["WC Change"], colnames=["OSM Change"]
    )
    del pred

    df_confusion_pandas_2 = pd.crosstab(
        actual, pred_2, rownames=["WC Change"], colnames=["OSM Built Up"]
    )
    del actual, pred_2

    # create CM Plot
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
    ax.set_xlabel("OSM Change", labelpad=10)
    ax.set_ylabel("WC Change", labelpad=10)
    border_linewidth = 1
    for _, spine in cm_map.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(border_linewidth)
    if not os.path.exists(CM_PATH):
        os.makedirs(CM_PATH)
    plt.title(
        f"Confusion Matrix stating if Change to Built-Up is in Accordance in both Datasets ({tilename})"
    )
    save_path_norm = pathlib.Path(f"{CM_PATH}/{tilename}_compare_change_area.png")
    plt.savefig(save_path_norm)
    plt.close()

    # calculate accordance. In Percent, how many Pixel with change to built up in WC are
    #  also change to built up in OSM
    try:
        accordance = (
            df_confusion_pandas.values[1][1]
            / (df_confusion_pandas.values[1][0] + df_confusion_pandas.values[1][1])
            * 100
        )
        # How many Pixels where change in WC happened are Built-Up in OSM
        accordance_2 = (
            df_confusion_pandas_2.values[0][1]
            / (df_confusion_pandas_2.values[0][0] + df_confusion_pandas_2.values[0][1])
            * 100
        )
    except:
        if (
            df_confusion_pandas.axes[0][0] == "Yes"
            and df_confusion_pandas.axes[1][0] == "No"
        ):
            accordance = 0
            accordance_2 = None
        elif (
            df_confusion_pandas.axes[0][0] == "Yes"
            and df_confusion_pandas.axes[1][0] == "Yes"
        ):
            accordance = 100
            accordance_2 = 100
        else:
            logging.error(f"Could not calculate accordance for tile {tilename}")
    try:
        try:
            osm_pixel_no_to_built = df_confusion_pandas["Yes"].sum()
        except:
            osm_pixel_no_to_built = 0
        try:
            wc_pixel_no_to_built = df_confusion_pandas.loc["Yes"].sum()
        except:
            wc_pixel_no_to_built = 0
    except:
        logging.error(
            f"Could not calculate number of pixels changed to built up for tile {tilename}"
        )
    del df_confusion_pandas, df_confusion_pandas_2

    return accordance, accordance_2, wc_pixel_no_to_built, osm_pixel_no_to_built


def adjusted_change_calculation(
    rasterpath_wc, comparepath_wc, rasterpath_osm, comparepath_osm
):
    tilename = get_tilename(rasterpath_wc)
    rasterdata_wc, comparedata_wc = get_rasterdata(rasterpath_wc, comparepath_wc)
    rasterdata_osm, comparedata_osm = get_rasterdata(rasterpath_osm, comparepath_osm)
    del comparepath_wc, rasterpath_osm, comparepath_osm

    # write 0 where class built-up was already present or is not present in newer dataset
    # write 1 where change to built-up happened but was not present in older OSM data
    changedata_wc = np.where(
        (rasterdata_wc == 50) & (comparedata_wc != 50) & (comparedata_osm != 50), 1, 0
    )
    changedata_wc_orig_vals = np.where((changedata_wc == 1), comparedata_wc, 0)

    resultfile = pathlib.Path(
        WC_COMP_PATH + f"{tilename}_wc_loss_of_nature_adjusted.tif"
    )
    loss_of_nature_vector(rasterpath_wc, changedata_wc_orig_vals, resultfile)

    wc_change_pixel = np.count_nonzero(changedata_wc)
    del (
        rasterpath_wc,
        rasterdata_wc,
        comparedata_wc,
        changedata_wc_orig_vals,
        resultfile,
    )

    changedata_osm = np.where((rasterdata_osm == 50) & (comparedata_osm != 50), 1, 0)

    # dort wo WC change und OSM nicht, schreibe neuen OSM value sonst 0
    osm_vals_where_wc_change = np.where(
        (changedata_wc == 1) & (changedata_osm == 0), rasterdata_osm, 0
    )
    unique, counts = np.unique(osm_vals_where_wc_change, return_counts=True)
    osm_where_wc_change = dict(zip(unique, counts))

    # remove value 0 (=where no WC change happened)
    osm_where_wc_change.pop(0, None)
    # get value how many pixels are no_data in OSM where WC has change
    no_osm_data_for_change = osm_where_wc_change[999]

    # remove all OSM_nodata for WC change pixels
    osm_where_wc_change.pop(999, None)
    # calculate how many osm pixels have a class but not built-up where WC has change to built-up
    other_osm_class_for_change = sum(osm_where_wc_change.values())
    del (
        rasterdata_osm,
        comparedata_osm,
        osm_vals_where_wc_change,
        unique,
        counts,
        osm_where_wc_change,
    )

    # write No where class built-up was already present or is not present in newer dataset
    # write Yes where change to built-up happened
    changedata_wc = np.where(changedata_wc == 1, "Yes", "No")
    changedata_osm = np.where(changedata_osm == 1, "Yes", "No")

    changedata_wc_masked = np.ma.masked_where(
        np.logical_and(changedata_wc == "No", changedata_osm == "No"), changedata_wc
    )
    changedata_osm_masked = np.ma.masked_where(
        np.logical_and(changedata_wc == "No", changedata_osm == "No"), changedata_osm
    )
    del changedata_wc, changedata_osm

    actual = np.nan_to_num(changedata_wc_masked.flatten(), nan=999)
    pred = np.nan_to_num(changedata_osm_masked.flatten(), nan=999)
    del changedata_wc_masked, changedata_osm_masked

    # create confusion Matrix using No. of Pixel
    df_confusion_pandas = pd.crosstab(
        actual, pred, rownames=["WC Change"], colnames=["OSM Change"]
    )
    del actual, pred

    # create CM Plot
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
    ax.set_xlabel("OSM Change", labelpad=10)
    ax.set_ylabel("WC Change", labelpad=10)
    border_linewidth = 1
    for _, spine in cm_map.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(border_linewidth)
    if not os.path.exists(CM_PATH):
        os.makedirs(CM_PATH)
    plt.title(
        f"Confusion Matrix stating if adjusted Change to Built-Up is matching in both Datasets ({tilename})"
    )
    save_path_norm = pathlib.Path(
        f"{CM_PATH}/{tilename}_compare_adjusted_change_area.png"
    )
    plt.savefig(save_path_norm)
    plt.close()

    try:
        matching_pixel = df_confusion_pandas.values[1][1]
    except:
        if (
            df_confusion_pandas.axes[0][0] == "Yes"
            and df_confusion_pandas.axes[1][0] == "No"
        ):
            matching_pixel = 0
        elif (
            df_confusion_pandas.axes[0][0] == "Yes"
            and df_confusion_pandas.axes[1][0] == "Yes"
        ):
            matching_pixel = wc_change_pixel
        else:
            logging.error(f"Could not calculate accordance for tile {tilename}")

    matching_percent = matching_pixel / wc_change_pixel * 100
    no_osm_data_for_change_percent = no_osm_data_for_change / wc_change_pixel * 100
    other_osm_class_for_change_percent = (
        other_osm_class_for_change / wc_change_pixel * 100
    )

    return (
        wc_change_pixel,
        matching_pixel,
        no_osm_data_for_change,
        other_osm_class_for_change,
        matching_percent,
        no_osm_data_for_change_percent,
        other_osm_class_for_change_percent,
    )


def main(compare_change=True, compare_wc=True, compare_wc_osm=True, compare_osm=True) -> None:
    """
    Main function to compare raster data between the years, between the data sets, and
    comparing the change each data set indicates
    :param compare_change: Bool to indicate if change between two years should be
    compared
    :param compare_wc: Bool to indicate if WorldCover data should be compared
    :param compare_wc_osm: Bool to indicate if WorldCover and OSM data should be
    compared per year
    :param compare_osm: Bool to indicate if OSM data should be compared
    :return: None
    """

    # set paths to raster files
    wc_datapath = pathlib.Path(TERRADIR + "Maps/")
    osm_datapath = pathlib.Path(OSMRASTER)

    # if a dir instead of file is set as input: use all geojson files in this dir
    if len(INFILES) == 0:
        for filename in os.listdir(INPUTDIR):
            if filename.endswith(".geojson"):
                INFILES.append(filename)

    # create a vector GDF which holds statistics for all raster
    statistics = gpd.GeoDataFrame()

    file_counter = 0
    # iterate over all input files
    for infile in INFILES:
        logging.info(f"Working on Inputfile {file_counter + 1} of {len(INFILES)}")

        # import geodata of AoIs
        gdf = import_geodata(INPUTDIR, infile)

        # Iterate over all features in the AoI file
        for index in gdf.index:
            logging.info(f"Working on Feature {index+1} of {len(gdf)}")

            # Get respective WC Raster to get extent from it in order to create a vector
            # feature from it and save statistics to it
            rastername_finder = (
                f"ESA_WorldCover_10m_2021_v200_f{file_counter:03}id{index:03}_Map.tif"
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

            # add attributes for feature from input file except do not overwrite geometry
            for i in range(len(gdf.loc[[index]].axes[1])):
                attr = gdf.loc[[index]].axes[1][i]
                if attr != "geometry":
                    feat_stats[attr] = gdf.loc[[index]][attr][index]

            # if the change per data set should be compared
            if compare_change:
                # get raster paths for WC and OSM for both years each
                rasterpath_wc = rasterpath
                comparepath_wc = get_wc_to_compare(rasterpath, wc_datapath)
                rastername_finder = f"f{file_counter:03}id{index:03}_2021.tif"
                rasterpath = pathlib.Path(osm_datapath / rastername_finder)
                rasterpath_osm = rasterpath
                comparepath_osm = get_osm_to_compare(rasterpath, osm_datapath)
                rasterpath = rasterpath_wc
                # calculate change statistics on full changeset
                (
                    feat_stats["change_accordance"],
                    feat_stats["change_accordance_2"],
                    feat_stats["wc_pixel_no_to_built"],
                    feat_stats["osm_pixel_no_to_built"],
                ) = compare_change_area(
                    rasterpath_wc, comparepath_wc, rasterpath_osm, comparepath_osm
                )
                # calculate change statistics on adjusted changeset
                (
                    feat_stats["wc_change_pixel"],
                    feat_stats["matching_pixel"],
                    feat_stats["no_osm_data_for_change"],
                    feat_stats["other_osm_class_for_change"],
                    feat_stats["matching_percent"],
                    feat_stats["no_osm_data_for_change_percent"],
                    feat_stats["other_osm_class_for_change_percent"],
                ) = adjusted_change_calculation(
                    rasterpath_wc, comparepath_wc, rasterpath_osm, comparepath_osm
                )

            # if change in WC between two years should be compared
            if compare_wc:
                # get raster path for file to be compared
                comparepath = get_wc_to_compare(rasterpath, wc_datapath)
                if comparepath is not None:
                    resultdir = WC_COMP_PATH
                    if not os.path.exists(resultdir):
                        os.makedirs(resultdir)
                    # calculate binary change
                    resultfile = pathlib.Path(
                        resultdir + f"{get_tilename(rasterpath)}_wc_binary_change.tif"
                    )
                    (
                        feat_stats["pixel_count"],
                        feat_stats["WC_Match_Pixel"],
                        feat_stats["WC_Match_Percent"],
                        feat_stats["WC_Match_Pixel_Agg"],
                        feat_stats["WC_Match_Percent_Agg"],
                    ) = detect_equality(rasterpath, comparepath, resultfile)[0:5]

                    # detect where classes have changed from another to built up
                    resultfile = pathlib.Path(
                        resultdir + f"{get_tilename(rasterpath)}_wc_loss_of_nature.tif"
                    )
                    aggregated_change_wc = detect_loss_of_nature(
                        rasterpath, comparepath, resultfile
                    )
                else:
                    logging.error(
                        f"Cannot find WC Raster to be compared with {rasterpath}"
                    )
            # if the WC and OSM datasets per year should be compared
            if compare_wc_osm:
                # iterate over both WC rasters, each representing one year
                for rasterpath in wc_datapath.rglob(
                    f"*f{file_counter:03}id{index:03}_Map.tif"
                ):
                    # get the respective OSM raster to compare against
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
                        # calculate completeness and accuracy of OSM dataset
                        results = detect_equality(rasterpath, comparepath, resultfile)
                        feat_stats[f"osm_acc_{tile_year}"] = results[2]
                        feat_stats[f"osm_acc_{tile_year}_no_nan"] = results[7]
                        feat_stats[f"osm_acc_agg_{tile_year}"] = results[4]
                        feat_stats[f"osm_acc_agg_{tile_year}_no_nan"] = results[8]
                        (
                            feat_stats[f"osm_nan_pixel_{tile_year}"],
                            feat_stats[f"osm_completeness_{tile_year}"],
                        ) = results[5:7]

                        # create confusion matrices
                        wc_data, osm_data = get_rasterdata(rasterpath, comparepath)
                        report = create_cm(
                            wc_data,
                            osm_data,
                            tilename=get_tilename(rasterpath),
                            year=get_tileyear(rasterpath),
                            aggregated=False,
                        )
                        if report is not None:
                            feat_stats[f"built-up_precision_{tile_year}"] = report.get(
                                "precision"
                            )
                            feat_stats[f"built-up_recall_{tile_year}"] = report.get(
                                "recall"
                            )

                        report_agg = create_cm(
                            wc_data,
                            osm_data,
                            tilename=get_tilename(rasterpath),
                            year=get_tileyear(rasterpath),
                            aggregated=True,
                        )
                        if report_agg is not None:
                            feat_stats[
                                f"built-up_precision_{tile_year}_agg"
                            ] = report_agg.get("precision")
                            feat_stats[
                                f"built-up_recall_{tile_year}_agg"
                            ] = report_agg.get("recall")
                        del wc_data, osm_data
                    else:
                        logging.error(
                            f"Cannot find OSM Raster to be compared with WC Raster {rasterpath}"
                        )
            # if the OSM datasets should be compared
            if compare_osm:
                # OSm raster for AoI and determine raster to be compared
                rastername_finder = f"f{file_counter:03}id{index:03}_2021.tif"
                rasterpath = pathlib.Path(osm_datapath / rastername_finder)
                comparepath = get_osm_to_compare(rasterpath, osm_datapath)
                if comparepath is not None:
                    resultdir = OSM_COMP_PATH
                    if not os.path.exists(resultdir):
                        os.makedirs(resultdir)
                    # calculate binary change
                    resultfile = pathlib.Path(
                        resultdir + f"{get_osmtile(rasterpath)}_osm_binary_change.tif"
                    )
                    (
                        feat_stats["pixel_count"],
                        feat_stats["OSM_Match_Pixel"],
                        feat_stats["OSM_Match_Percent"],
                        feat_stats["OSM_Match_Pixel_Agg"],
                        feat_stats["OSM_Match_Percent_Agg"],
                    ) = detect_equality(rasterpath, comparepath, resultfile)[0:5]
                    # detect where classes have changed from another to built up
                    resultfile = pathlib.Path(
                        resultdir + f"{get_osmtile(rasterpath)}_osm_loss_of_nature.tif"
                    )
                    aggregated_change_osm = detect_loss_of_nature(
                        rasterpath, comparepath, resultfile
                    )
                else:
                    logging.error(
                        f"Cannot find OSM Raster to be compared with {rasterpath}"
                    )
            if compare_wc and compare_osm:
                # create confusion matrix whether change to built-up between WC & OSM
                # agrees
                create_cm(
                    aggregated_change_wc,
                    aggregated_change_osm,
                    tilename=get_osmtile(rasterpath),
                    change_cm=True,
                )

            # add data frame of this AoI with all added statistics to data frame of all
            # AoIs
            statistics = pd.concat([statistics, feat_stats], ignore_index=True)

    # write data frame of all AoIs with added statistics as geojson file
    stats_outpath = pathlib.Path(f"{COMP_PATH}/statistics.geojson")
    with open(stats_outpath, "w") as file:
        file.write(statistics.to_json())


if __name__ == "__main__":
    # warnings.filterwarnings("error")
    with open(LOGGCONFIG, "r") as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)

    main(compare_change=True, compare_wc=True, compare_wc_osm=True, compare_osm=True)
