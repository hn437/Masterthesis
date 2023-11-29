import asyncio
import logging
import logging.config

import yaml

import download_worldcover_data
import compare_rasters
import download_osm_data
import calculate_correlation
from config import LOGGCONFIG


async def main():
    logging.info("Running Script 1: Downloading WC Data")
    download_worldcover_data.main()
    logging.info("Finished Script 1")
    logging.info("Running Script 2: Downloading OSM Data")
    await download_osm_data.main()
    logging.info("Finished Script 2")
    logging.info("Running Script 3: Comparing Data")
    compare_rasters.main(
        compare_change=True, compare_wc=True, compare_wc_osm=True, compare_osm=True
    )
    logging.info("Finished Script 3")
    logging.info("Running Script 4: Calculating Correlations")
    calculate_correlation.main()
    logging.info("Finished Script 4")
    logging.info("Analysis Finished!")


if __name__ == "__main__":
    with open(LOGGCONFIG, "r") as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)

    asyncio.run(main())
