import os
import sys

import geopandas as gpd
from terracatalogueclient import Catalogue

from config import INFILES, INPUTDIR, PASSWORD, TERRADIR, USERNAME


def import_geodata(inputdir, infile):
    if infile is not None:
        path_to_file = os.path.join(inputdir, infile)
        df = gpd.read_file(path_to_file)

        if df.crs != 4326:
            df = df.to_crs(4326)

    return df


def main():
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

        for index in gdf.index:
            feature = gdf.loc[[index]]
            geom = feature.geometry[index]
            products = catalogue.get_products(
                "urn:eop:VITO:ESA_WorldCover_10m_2021_V2", geometry=geom
            )

            # download the products to the given directory
            if not os.path.exists(TERRADIR):
                os.makedirs(TERRADIR)

            sys.stdout.write(
                f"\r\tDownloading WorldCover data for feature {index+1} of {len(gdf)}"
            )
            catalogue.download_products(products, TERRADIR, force=True)

        file_counter += 1

    print(
        f"\n\nFinished downloading WorldCover Data for all Files and Features to "
        f"directory {TERRADIR}"
    )


if __name__ == "__main__":
    main()
