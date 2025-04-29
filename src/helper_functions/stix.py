import pandas as pd
from stixdcpy import auxiliary as aux
from stixdcpy.quicklook import LightCurves
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')


def get_flarelist(path_to_flarelist):
    try:
        return pd.read_csv(path_to_flarelist)
    except FileNotFoundError:
        logging.info(f"Flarelist file not found at {path_to_flarelist}.")


def load_stix_light_curve(start_utc, end_utc):
    try:
        return LightCurves.from_sdc(start_utc, end_utc, ltc=True)
    except Exception as e:
        logging.info(f"Error loading light curves: {e}")


def get_position(start, end):
    try:
        return aux.Ephemeris.from_sdc(start_utc=start, end_utc=end, steps=1)
    except Exception as e:
        logging.info(f"Error loading position data: {e}")
