import pyvo
import logging
import numpy as np
from dateutil import parser
from helper_functions.utils import safe_parse_time


def get_mwa_metadata(start_time=None, end_time=None, obs_ids=None):
    tap_service = pyvo.dal.TAPService("http://vo.mwatelescope.org/mwa_asvo/tap")

     # define the query based on the input parameters
    if obs_ids is not None:
        ids_formatted = ', '.join(f"'{id}'" for id in obs_ids)
        query = f"SELECT * FROM mwa.observation WHERE obs_id IN ({ids_formatted})"
    elif start_time is not None and end_time is not None:
        query = f"""
        SELECT * FROM mwa.observation
        WHERE stoptime_utc >= '{format_time_for_mwa(start_time)}'
        AND starttime_utc <= '{format_time_for_mwa(end_time)}'
        """
    else:
        raise ValueError("Invalid parameters. Provide either 'obs_id' or both 'start_time' and 'end_time'.")

     # execute the query
    result = tap_service.search(query)
    mwa_metadata = result.to_table().to_pandas().sort_values(by='starttime_utc').reset_index(drop=True)
    logging.info(f"Number of found observations is {len(mwa_metadata)}")
    return mwa_metadata


def format_time_for_mwa(time_str):
    dt = parser.parse(time_str)
     # format the datetime to the desired format, cutting off milliseconds to 3 digits
    formatted_time = dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    return formatted_time


def get_mwa_light_curve(spectrogram):
    return np.ma.sum(spectrogram, axis=0) if spectrogram is not None else None


def estimate_time_resolution(spectrograms, times):
    """
    estimates time resolution in seconds from spectrogram shapes and time ranges
    """
    resolutions = []
    for dspec, (start, end) in zip(spectrograms, times):
        spec = dspec['spec']
        start = safe_parse_time(start)
        end = safe_parse_time(end)

        duration = (end - start).total_seconds()
        if spec.shape[1] > 0:
            res = duration / spec.shape[1]
            resolutions.append(res)

    if resolutions:
        return round(np.mean(resolutions))
    return None
