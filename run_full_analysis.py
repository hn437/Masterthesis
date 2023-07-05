import asyncio
import logging
import logging.config

import yaml

from config import LOGGCONFIG

with open(LOGGCONFIG, "r") as f:
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)
logger = logging.getLogger(__name__)

import compare_rasters
import download_osm_data
import download_worldcover_data


async def main():
    logger.info(f"Running Script 1: Downloading WC Data")
    download_worldcover_data.main()
    logger.info(f"Finished Script 1")
    logger.info(f"Running Script 2: Downloading OSM Data")
    await download_osm_data.main()
    logger.info(f"Finished Script 2")
    logger.info(f"Running Script 3: Comparing Data")
    compare_rasters.main(compare_wc=True, compare_wc_osm=True, compare_osm=True)
    logger.info(f"Finished Script 3")
    logger.info(f"Analysis Finished!")


if __name__ == "__main__":
    asyncio.run(main())
