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


def calculate_with_nominal(continuous_data, nominal_data):
    return stats.pointbiserialr(nominal_data, continuous_data)


def calculate_with_scalar(accordance, continuous_data):
    return stats.pearsonr(accordance, continuous_data)


def calculate_correlation(df):
    accordance = df['change_accordance']
    names = []
    correlations = []
    p_values = []

    for column in (['University']):
        correlation, p_value = calculate_with_nominal(accordance, df[column])
        names.append(column)
        correlations.append(correlation)
        p_values.append(p_value)

    for column in ([]):
        correlation, p_value = calculate_with_scalar(accordance, df[column])
        names.append(column)
        correlations.append(correlation)
        p_values.append(p_value)

    return names, correlations, p_values


def create_correlation_table(names, correlations, p_values):
    # create table of all correlations
    data = {
        'correlation': correlations,
        'p_value': p_values
    }
    correlation_table = pd.DataFrame(data, index=names)

    # save as Excel file
    outfile = pathlib.Path(f"{COMP_PATH}/correlations.xlsx")
    correlation_table.to_excel(outfile, index=True)

    # create heatmap
    colormaps = ['Blues', 'OrRd']

    fig, axes = plt.subplots(1, 2, figsize=(20, 15))  # 1 row, 2 columns for two subplots

    # Create a heatmap for each column
    sns.heatmap(correlation_table[['correlation']], annot=True, cmap=colormaps[0], ax=axes[0], xticklabels=False, cbar_kws={'location': 'bottom'})
    sns.heatmap(correlation_table[['p_value']], annot=True, cmap=colormaps[1], ax=axes[1], xticklabels=False, yticklabels=False, cbar_kws={'location': 'bottom'})

    axes[0].set_title('Correlation')
    axes[1].set_title('P-Value')

    plt.suptitle(
        f"Correlation between Accordance of Change to Built-Up and Other Variables"
    )
    plt.tight_layout()

    if not os.path.exists(COMP_PATH):
        os.makedirs(COMP_PATH)
    save_path_norm = pathlib.Path(f"{COMP_PATH}/correlations.png")
    plt.savefig(save_path_norm)


def main():
    infile = "statistics.geojson"
    gdf = import_geodata(COMP_PATH, infile)
    names, correlations, p_values = calculate_correlation(gdf)

    create_correlation_table(names, correlations, p_values)



if __name__ == "__main__":
    # warnings.filterwarnings("error")
    with open(LOGGCONFIG, "r") as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)

    main()

    #TODO: Logging, Normalverteilung testen