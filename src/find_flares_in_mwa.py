import os
import time
import pandas as pd
import pytz
import pyvo
from datetime import timedelta, time
from astral import LocationInfo
from astral.sun import sun
from sunpy.coordinates.sun import earth_distance
from sunpy.time import parse_time
import astropy.units as u
from astropy.constants import c


def main():
    """
    runs the main workflow to match stix flares with mwa observations,
    applies time correction, and saves filtered flare data to disk
    """
    stix_flares_path = "../info/STIX_flarelist_w_locations_20210214_20250228_version1_python.csv"
    mwa_data = load_and_preprocess_mwa_metadata()
    mwa_location = LocationInfo("MWA", "Australia", "Australia/Perth", latitude=-26.7033, longitude=116.6708)

    for use_time_correction in [True, False]:
        print(f"Time correction: {use_time_correction}")
        stix_data = load_and_preprocess_stix_data(stix_flares_path, use_time_correction)

        save_path = (
            '../info/flares_recorded_by_mwa_with_time_correction.csv'
            if use_time_correction else
            '../info/flares_recorded_by_mwa_no_time_correction.csv'
        )
        
        flares_df, num_samples = analyze_and_filter_flares(stix_data, mwa_data, mwa_location)
        flares_df = add_mwa_project_and_obs_ids(flares_df, mwa_data)
        flares_df.to_csv(save_path, index=False)
        
        print_summary(num_samples)


def analyze_and_filter_flares(stix_data, mwa_data, mwa_location):
    flares_df, num_samples = analyze_flare_data(stix_data, mwa_data, mwa_location)
    flares_df = find_flares_with_overlap(flares_df, overlap_percentage=1)
    flares_df = sort(flares_df, sort_values_list=['overlap_percentage', 'GOES_class'])
    flares_df = flares_df[[
        'flare_id', 'GOES_class', 'stix_start_UTC', 'stix_end_UTC',
        'mwa_start_UTC', 'mwa_end_UTC', 'flare_duration_sec', 'overlap_percentage']].reset_index(drop=True)
    return flares_df, num_samples


def print_summary(num_samples):
    for key, value in num_samples.items():
        print(f"{key}: {value}")


def load_and_preprocess_mwa_metadata(save=False, path_to_save="../info"):
    """
    loads mwa metadata using pyvo and preprocesses datetime fields
    """
    tap_service = pyvo.dal.TAPService("http://vo.mwatelescope.org/mwa_asvo/tap")
    query = """SELECT * FROM mwa.observation WHERE starttime_utc > '2021-02-14T00:00:00Z'"""
    result = tap_service.search(query)

    mwa = result.to_table().to_pandas().sort_values(by='starttime_utc').reset_index(drop=True)

    if save:
        mwa.to_csv(os.path.join(path_to_save, "mwa_metadata.csv"), index=False)

    for col in ['starttime_utc', 'stoptime_utc']:
        mwa[col] = pd.to_datetime(mwa[col]).apply(
            lambda x: x.tz_convert('UTC') if x.tzinfo else x.tz_localize('UTC')
        )
    return mwa


def load_and_preprocess_stix_data(filepath, use_time_correction):
    """
    loads stix flare data and applies optional light travel time correction
    """
    stix = pd.read_csv(filepath)
    stix['start_UTC'] = pd.to_datetime(stix['start_UTC']).dt.tz_localize('UTC')
    stix['end_UTC'] = pd.to_datetime(stix['end_UTC']).dt.tz_localize('UTC')
    stix = stix[stix['visible_from_earth']].reset_index(drop=True)

    stix['time_correction'] = calculate_time_correction(stix)

    if use_time_correction:
        stix['mwa_start_UTC'] = stix['start_UTC'] + pd.to_timedelta(stix['time_correction'], unit='s')
        stix['mwa_end_UTC'] = stix['end_UTC'] + pd.to_timedelta(stix['time_correction'], unit='s')
    else:
        stix['mwa_start_UTC'] = stix['start_UTC']
        stix['mwa_end_UTC'] = stix['end_UTC']

    stix['flare_duration_sec'] = (stix['mwa_end_UTC'] - stix['mwa_start_UTC']).dt.total_seconds()
    return stix


def calculate_time_correction(stix_data):
    """
    calculates time correction (in seconds) based on light travel difference
    between solar orbiter and earth
    """
    time = parse_time(stix_data['peak_UTC']).datetime
    earth_distance_AU = u.Quantity(earth_distance(time), u.AU)
    stix_distance_AU = u.Quantity(stix_data['solo_position_AU_distance'], u.AU)
    diff = earth_distance_AU - stix_distance_AU
    return (diff.to(u.m) / c).value


def analyze_flare_data(stix, mwa_data, mwa_location):
    """
    calculates overlap between stix flares and mwa observations
    """
    time_overlap_data = []
    num_samples = {f'matching {i*10}-{(i+1)*10}%': 0 for i in range(10)}
    num_samples['num_of_matching_observations'] = 0

    for _, flare_row in stix.iterrows():
        flare_start, flare_end = flare_row['mwa_start_UTC'], flare_row['mwa_end_UTC']
        flare_duration = flare_row['flare_duration_sec']
        overlap = calculate_overlap(mwa_data, flare_start, flare_end, mwa_location)
        overlap_percentage = int(100 * overlap.total_seconds() / flare_duration)

        time_overlap_data.append({
            'flare_id': flare_row['flare_id'],
            'GOES_class': flare_row['GOES_class_time_of_flare'],
            'stix_start_UTC': flare_row['start_UTC'],
            'stix_end_UTC': flare_row['end_UTC'],
            'mwa_start_UTC': flare_row['mwa_start_UTC'],
            'mwa_end_UTC': flare_row['mwa_end_UTC'],
            'flare_duration_sec': int(flare_duration),
            'overlap_duration_sec': int(overlap.total_seconds()),
            'overlap_percentage': overlap_percentage
        })

        if overlap_percentage > 0:
            num_samples['num_of_matching_observations'] += 1
        for i in range(10):
            if i * 10 < overlap_percentage <= (i + 1) * 10:
                num_samples[f'matching {i*10}-{(i+1)*10}%'] += 1

    return pd.DataFrame(time_overlap_data), num_samples


def calculate_overlap(mwa_data, flare_start, flare_end, mwa_location):
    """
    calculates total overlap duration during daylight hours at MWA
    """
    mwa_relevant = mwa_data[
        (mwa_data['starttime_utc'] <= flare_end) &
        (mwa_data['stoptime_utc'] >= flare_start)
    ]
    total_overlap = timedelta(seconds=0)

    for _, mwa_row in mwa_relevant.iterrows():
        overlap_start = max(flare_start, mwa_row['starttime_utc'])
        overlap_end = min(flare_end, mwa_row['stoptime_utc'])

        times = sun(mwa_location.observer, date=overlap_start.date(), tzinfo=pytz.UTC)
        adjusted_sunrise = times['sunrise'] - timedelta(days=1) if time(0, 0) <= overlap_start.time() < time(12, 0) else times['sunrise']
        adjusted_sunset = times['sunset'] + timedelta(days=1) if time(17, 0) <= overlap_start.time() <= time(23, 59) else times['sunset']

        total_overlap_start = max(overlap_start, adjusted_sunrise)
        total_overlap_end = min(overlap_end, adjusted_sunset)

        if total_overlap_start < total_overlap_end:
            total_overlap += total_overlap_end - total_overlap_start

    return total_overlap


def find_flares_with_overlap(df, overlap_percentage):
    """
    filters dataframe for flares with sufficient overlap percentage
    """
    return df[df['overlap_percentage'] >= overlap_percentage].reset_index(drop=True)


def sort(df, sort_values_list=['overlap_percentage', 'goes_numeric']):
    """
    sorts flares based on overlap and GOES class
    """
    df['goes_numeric'] = df['GOES_class'].apply(goes_class_to_numeric)
    return df.sort_values(by=sort_values_list, ascending=[False, False], ignore_index=True)


def goes_class_to_numeric(goes_class):
    """
    converts GOES flare class (e.g., M5.6) into a numeric value
    """
    if pd.isna(goes_class):
        return -1
    scale = {'A': 1e-8, 'B': 1e-7, 'C': 1e-6, 'M': 1e-5, 'X': 1e-4}
    try:
        prefix = goes_class[0].upper()
        magnitude = float(goes_class[1:]) if len(goes_class) > 1 else 1.0
        return scale[prefix] * magnitude
    except (KeyError, ValueError):
        return -1


def add_mwa_project_and_obs_ids(flares_df, mwa_data):
    """
    attaches mwa project and observation ids to each flare
    """
    flares_df['projectids'] = [[] for _ in range(len(flares_df))]
    flares_df['obs_ids'] = [[] for _ in range(len(flares_df))]

    for idx, row in flares_df.iterrows():
        flare_start = pd.to_datetime(row['mwa_start_UTC'])
        flare_end = pd.to_datetime(row['mwa_end_UTC'])
        matching_data = mwa_data[(mwa_data['starttime_utc'] <= flare_end) & (mwa_data['stoptime_utc'] >= flare_start)]
        flares_df.at[idx, 'projectids'] = matching_data['projectid'].tolist()
        flares_df.at[idx, 'obs_ids'] = matching_data['obs_id'].tolist()

    return flares_df


if __name__ == "__main__":
    main()
