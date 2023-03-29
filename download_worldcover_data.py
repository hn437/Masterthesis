import os
import pathlib
import shutil
import sys

import geopandas as gpd
from terracatalogueclient import Catalogue

from config import INFILES, INPUTDIR, PASSWORD, TERRADIR, USERNAME


def import_geodata(inputdir: str, infile: str) -> gpd.GeoDataFrame:
    if infile is not None:
        path_to_file = pathlib.Path(inputdir, infile)
        df = gpd.read_file(path_to_file)

        if df.crs != 4326:
            df = df.to_crs(4326)

    return df


def download_terrascope_data(directory: pathlib.Path) -> None:
    # Authenticate to the Terrascope platform and create catalogue object
    catalogue = Catalogue().authenticate_non_interactive(
        username=USERNAME, password=PASSWORD
    )

    if len(INFILES) == 0:
        for filename in os.listdir(INPUTDIR):
            if filename.endswith(".geojson"):
                INFILES.append(filename)

    file_counter = 1
    for infile in INFILES:
        print(f"\nWorking on Inputfile {file_counter} of {len(INFILES)}")
        # import geodata of extent to be imported from WorldCover
        gdf = import_geodata(INPUTDIR, infile)

        if not os.path.exists(directory):
            os.makedirs(directory)

        for index in gdf.index:
            feature = gdf.loc[[index]]
            geom = feature.geometry[index]

            products_2020 = catalogue.get_products(
                "urn:eop:VITO:ESA_WorldCover_10m_2020_V1", geometry=geom
            )
            products_2021 = catalogue.get_products(
                "urn:eop:VITO:ESA_WorldCover_10m_2021_V2", geometry=geom
            )

            # download the products to the given directory
            sys.stdout.write(
                f"\r\tDownloading WorldCover data for feature {index+1} of {len(gdf)}"
            )
            catalogue.download_products(products_2020, directory, force=True)
            catalogue.download_products(products_2021, directory, force=True)

        file_counter += 1

    print(
        f"\n\nFinished downloading WorldCover Data for all Files and Features to "
        f"directory {directory}\n\n"
    )


def clean_terradata(scratch_dir: pathlib.Path) -> None:
    print(f"Extracting WorldCover Mapdata and Quality Data")

    maps_dir = TERRADIR + "Maps/"
    quality_dir = TERRADIR + "Quality/"
    if not os.path.exists(maps_dir):
        os.makedirs(maps_dir)
    if not os.path.exists(quality_dir):
        os.makedirs(quality_dir)

    for file in scratch_dir.rglob("*_Map.tif"):
        shutil.move(file, maps_dir + file.name)

    for file in scratch_dir.rglob("*_InputQuality.tif"):
        shutil.move(file, quality_dir + file.name)

    shutil.rmtree(scratch_dir)

    print(f"Finished! All Mapdata located in {maps_dir}, Qualitydata in {quality_dir}")


def main():
    path_terradir_scratch = pathlib.Path(TERRADIR + "scratch/")

    download_terrascope_data(path_terradir_scratch)
    clean_terradata(path_terradir_scratch)


if __name__ == "__main__":
    main()
