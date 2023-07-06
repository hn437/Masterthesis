import asyncio
import logging
import logging.config

import yaml

import compare_rasters
import download_osm_data
import download_worldcover_data
from config import LOGGCONFIG


async def main():
    logging.info(f"Running Script 1: Downloading WC Data")
    download_worldcover_data.main()
    logging.info(f"Finished Script 1")
    logging.info(f"Running Script 2: Downloading OSM Data")
    await download_osm_data.main()
    logging.info(f"Finished Script 2")
    logging.info(f"Running Script 3: Comparing Data")
    compare_rasters.main(compare_wc=True, compare_wc_osm=True, compare_osm=True)
    logging.info(f"Finished Script 3")
    logging.info(f"Analysis Finished!")


if __name__ == "__main__":
    with open(LOGGCONFIG, "r") as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)

    asyncio.run(main())
