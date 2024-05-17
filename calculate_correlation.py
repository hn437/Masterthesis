import logging
import logging.config
import os
import pathlib
from scipy import stats

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import yaml

from config import (
    COMP_PATH,
    LOGGCONFIG,
)
from download_worldcover_data import import_geodata


def calculate_with_nominal(continuous_data, nominal_data) -> tuple[float, float]:
    """
    Calculates a point biserial correlation coefficient and the associated p-value
    between an accordance value and a nominal data type characteristic of the AoIs.
    :param continuous_data: accordance values
    :param nominal_data: values of the nominal data
    :return: tuple of the calculated correlation & p_value value
    """
    return stats.pointbiserialr(nominal_data, continuous_data)


def calculate_with_scalar(
    accordance, continuous_data
) -> tuple[tuple[float, float], str]:
    """
    Calculates the correlation between an accordance value and an ordinal data type
    :param accordance: accordance values
    :param continuous_data: values of the ordinal data
    :return: tuple of the calculated correlation & p_value value as well as the type of
    test
    """
    # check for Gaussian Distribution with Shapiro-Wilk Test
    statistic, p = stats.shapiro(continuous_data)
    # choose the appropriate correlation test
    if p > 0.05:
        # if Gaussian Distribution exists in the data, use Pearson correlation
        correlation = stats.pearsonr(accordance, continuous_data)
        corr_type = "Pearson Correlation"
    else:
        # if no Gaussian Distribution exists in the data, use Spearman correlation
        correlation = stats.spearmanr(accordance, continuous_data)
        corr_type = "Spearman Correlation"
    return correlation, corr_type


def calculate_correlation(df, accordance_no) -> tuple[list, list, list, list]:
    """
    Calculate the correlation between the accordance value and the characteristics of
     the AoIs
    :param df: GeoDataFrame containing the AoIs and their characteristics and Statistics
    :param accordance_no: defines which accordance value should be used for the test
    :return: tuple of lists containing the names of the characteristics, the correlation
     values, the p-values and the type of correlation test
    """
    if accordance_no == 1:
        accordance = df["change_accordance"]
    elif accordance_no == 2:
        accordance = df["change_accordance_2"]
    elif accordance_no == 3:
        accordance = df["matching_percent"]
    else:
        logging.error("Accordance out of Range")

    names = []
    correlations = []
    p_values = []
    corr_test_type = []

    # calculate correlation for nominal data
    for column in ["University"]:
        correlation, p_value = calculate_with_nominal(accordance, df[column])
        # append results to lists
        names.append(column)
        correlations.append(correlation)
        p_values.append(p_value)
        corr_test_type.append("point biserial correlation")

    # calculate correlation for ordinal data
    for column in [
        "Population",
        "X-Coordinate",
        "Y-Coordinate",
        "Disposable Income",
        "osm_completeness_2021",
        "osm_acc_agg_2021_no_nan",
    ]:
        result = calculate_with_scalar(accordance, df[column])
        correlation, p_value = result[0]
        corr_type = result[1]
        # append results to lists
        names.append(column)
        correlations.append(correlation)
        p_values.append(p_value)
        corr_test_type.append(corr_type)

    return names, correlations, p_values, corr_test_type


def create_correlation_table(names, correlations, p_values, accordance_no) -> None:
    """
    creates a heatmap table of the correlation results and saves it
    :param names: names of characteristics analysed
    :param correlations: correlation values caluclated
    :param p_values: associated p-values to correlation values
    :param accordance_no: Number of accordance value used for the test
    :return: None
    """
    # create table of all correlations
    data = {"correlation": correlations, "p_value": p_values}
    correlation_table = pd.DataFrame(data, index=names)

    # define file stem and part of title depending on the accordance value used
    if accordance_no == 1:
        file_stem = "correlations_acc"
        title_insert = "Accordance"
    elif accordance_no == 2:
        file_stem = "correlations_acc_2"
        title_insert = "Accordance 2"
    elif accordance_no == 3:
        file_stem = "correlations_adjusted"
        title_insert = "Adjusted Match"
    else:
        logging.error("Accordance out of Range")

    # save Result table as Excel file
    outfile = pathlib.Path(f"{COMP_PATH}/{file_stem}.xlsx")
    correlation_table.to_excel(outfile, index=True)

    # create heatmap
    colormaps = ["Blues", "OrRd"]

    fig, axes = plt.subplots(
        1, 2, figsize=(20, 15)
    )  # 1 row, 2 columns for two subplots

    # Create a heatmap for each column (correlation and p-value
    sns.heatmap(
        correlation_table[["correlation"]],
        annot=True,
        cmap=colormaps[0],
        ax=axes[0],
        xticklabels=False,
        cbar_kws={"location": "bottom"},
    )
    sns.heatmap(
        correlation_table[["p_value"]],
        annot=True,
        cmap=colormaps[1],
        ax=axes[1],
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"location": "bottom"},
    )

    # Set titles for the subplots
    axes[0].set_title("Correlation")
    axes[1].set_title("P-Value")

    plt.suptitle(
        f"Correlation between {title_insert} of Change to Built-Up and Other Variables"
    )
    plt.tight_layout()

    # save the heatmap
    if not os.path.exists(COMP_PATH):
        os.makedirs(COMP_PATH)
    save_path_norm = pathlib.Path(f"{COMP_PATH}/{file_stem}.png")
    plt.savefig(save_path_norm)


def main() -> None:
    """
    Calculate correlations between accordance values and characteristics of the AoIs
    :return: None
    """

    # Load the statistics GeoDataFrame which contains the AoIs, accordance values and
    # the characteristics of the AoIs
    infile = "statistics.geojson"
    gdf = import_geodata(COMP_PATH, infile)
    # Calculate the correlations for each accordance values
    for accordance_no in range(1, 4):
        logging.info(f"Calculating correlations for accordance no. {accordance_no}.")
        # calculate correlation results
        names, correlations, p_values, corr_test_type = calculate_correlation(
            gdf, accordance_no
        )
        logging.info("Correlation calculation finished. Create Table of results.")
        # create a table of all correlation results and save it
        create_correlation_table(names, correlations, p_values, accordance_no)
        logging.info("Table created and saved.")


if __name__ == "__main__":
    # warnings.filterwarnings("error")
    with open(LOGGCONFIG, "r") as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)

    main()
