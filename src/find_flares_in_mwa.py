import os
import time
import pandas as pd
import pytz
from datetime import timedelta, time
from astral import LocationInfo
from astral.sun import sun
from sunpy.coordinates.sun import earth_distance
from sunpy.time import parse_time
import astropy.units as u
from astropy.constants import c


path_to_data = "../data"

def main():
    mwa_data_name = "mwa_metadata.csv"
    stix_data_name = "STIX_flarelist_w_locations_20210214_20230901_version1.csv"
    output_path = "../data/flares_recorded_by_mwa.csv"    

    mwa_data = load_and_preprocess_mwa_data(os.path.join(path_to_data, mwa_data_name))
    stix_data = load_and_preprocess_stix_data(os.path.join(path_to_data, stix_data_name))

    mwa_filtered = filter_mwa_within_stix_timeframe(mwa_data, stix_data)
    mwa_location = LocationInfo("MWA", "Australia", "Australia/Perth", latitude=-26.7033, longitude=116.6708)

    flares_df, num_samples = analyze_flare_data(stix_data, mwa_filtered, mwa_location)
    flares_df = find_flares_with_overlap(flares_df, overlap_percentage=1)
    flares_df = flares_df[['flare_id', 'flare_class_GOES', 'flare_start_UTC', 'flare_end_UTC', 'flare_start_UTC_corrected', 'flare_end_UTC_corrected', 'flare_duration_sec', 'total_overlap_percentage']].reset_index(drop=True)
    flares_df = flares_df.sort_values(by='total_overlap_percentage', ascending=False, ignore_index=True)
    flares_df = add_mwa_project_and_obs_ids(flares_df, mwa_data)

    flares_df.to_csv(output_path, index=False)
    for key, value in num_samples.items():
        print(f"{key}: {value}")


def load_and_preprocess_mwa_data(filepath):
    mwa = pd.read_csv(filepath, low_memory=False)
    mwa['starttime_utc'] = pd.to_datetime(mwa['starttime_utc']).apply(
        lambda x: x.tz_convert('UTC') if x.tzinfo else x.tz_localize('UTC')
    )
    mwa['stoptime_utc'] = pd.to_datetime(mwa['stoptime_utc']).apply(
        lambda x: x.tz_convert('UTC') if x.tzinfo else x.tz_localize('UTC')
    )
    return mwa


def load_and_preprocess_stix_data(filepath):
    stix = pd.read_csv(filepath)
    stix['start_UTC'] = pd.to_datetime(stix['start_UTC']).dt.tz_localize('UTC')
    stix['end_UTC'] = pd.to_datetime(stix['end_UTC']).dt.tz_localize('UTC')
    stix = stix[stix['visible_from_earth']].reset_index(drop=True)
    time_in_seconds = calculate_time_correction(stix)
    stix['time_correction'] = time_in_seconds
    stix['corrected_start_UTC'] = stix['start_UTC'] + pd.to_timedelta(stix['time_correction'], unit='s')
    stix['corrected_end_UTC'] = stix['end_UTC'] + pd.to_timedelta(stix['time_correction'], unit='s')
    stix['flare_duration_sec'] = (stix['corrected_end_UTC'] - stix['corrected_start_UTC']).dt.total_seconds()
    return stix


def calculate_time_correction(stix_data):
    # the time you want to calculate it at
    time = parse_time(stix_data['peak_UTC']).datetime

    # ensure earth_distance(time) is an Astropy Quantity in AU
    earth_distance_AU = u.Quantity(earth_distance(time), u.AU)

    # ensure STIX data is a compatible Quantity
    stix_distance_AU = u.Quantity(stix_data['solo_position_AU_distance'], u.AU)

    # get the difference
    diff = earth_distance_AU - stix_distance_AU

    # convert to meters and divide by the speed of light
    time_in_seconds = (diff.to(u.m) / c)
    return time_in_seconds.value # .value removes units (seconds) from the array


def filter_mwa_within_stix_timeframe(mwa, stix):
    stix_min_time = stix['corrected_start_UTC'].min()
    stix_max_time = stix['corrected_end_UTC'].max()
    return mwa[(mwa['stoptime_utc'] >= stix_min_time) & (mwa['starttime_utc'] <= stix_max_time)]


def calculate_overlap(mwa_filtered, flare_start, flare_end, mwa_location):
    mwa_relevant = mwa_filtered[
        (mwa_filtered['starttime_utc'] <= flare_end) & 
        (mwa_filtered['stoptime_utc'] >= flare_start)
    ]
    total_overlap, daytime_overlap = timedelta(seconds=0), timedelta(seconds=0)
    
    for _, mwa_row in mwa_relevant.iterrows():
        overlap_start, overlap_end = max(flare_start, mwa_row['starttime_utc']), min(flare_end, mwa_row['stoptime_utc'])
        total_overlap += overlap_end - overlap_start

        observation_date = overlap_start.date()
        times = sun(mwa_location.observer, date=observation_date, tzinfo=pytz.UTC)
        adjusted_sunrise = times['sunrise'] - timedelta(days=1) if time(0, 0) <= overlap_start.time() < time(12, 0) else times['sunrise']
        adjusted_sunset = times['sunset'] + timedelta(days=1) if time(17, 0) <= overlap_start.time() <= time(23, 59) else times['sunset']
        
        daylight_overlap_start = max(overlap_start, adjusted_sunrise)
        daylight_overlap_end = min(overlap_end, adjusted_sunset)
        
        if daylight_overlap_start < daylight_overlap_end:
            daytime_overlap += daylight_overlap_end - daylight_overlap_start

    return total_overlap, daytime_overlap


def analyze_flare_data(stix, mwa_filtered, mwa_location):
    time_overlap_data = []
    num_samples = {f'matching {i*10}-{(i+1)*10}%': 0 for i in range(10)}
    num_samples['num_of_matching_observations'] = 0

    for _, flare_row in stix.iterrows():
        flare_start, flare_end, flare_duration = flare_row['corrected_start_UTC'], flare_row['corrected_end_UTC'], flare_row['flare_duration_sec']
        total_overlap, daytime_overlap = calculate_overlap(mwa_filtered, flare_start, flare_end, mwa_location)
        
        overlap_percentage = int(100 * total_overlap.total_seconds() / flare_duration)
        daytime_overlap_percentage = int(100 * daytime_overlap.total_seconds() / flare_duration)

        time_overlap_data.append({
            'flare_id': flare_row['flare_id'],
            'flare_class_GOES': flare_row['GOES_class_time_of_flare'],
            'flare_start_UTC': flare_row['start_UTC'],
            'flare_end_UTC': flare_row['end_UTC'],
            'flare_start_UTC_corrected': flare_row['corrected_start_UTC'],
            'flare_end_UTC_corrected': flare_row['corrected_end_UTC'],
            'flare_duration_sec': int(flare_duration),
            'overlap_with_mwa_duration_sec': int(total_overlap.total_seconds()),
            'overlap_with_mwa_duration_percentage': overlap_percentage,
            'total_overlap_duration_sec': int(daytime_overlap.total_seconds()),
            'total_overlap_percentage': daytime_overlap_percentage
        })

        if daytime_overlap_percentage > 0:
            num_samples['num_of_matching_observations'] += 1
        for i in range(10):
            if i * 10 < daytime_overlap_percentage <= (i + 1) * 10:
                num_samples[f'matching {i*10}-{(i+1)*10}%'] += 1

    return pd.DataFrame(time_overlap_data), num_samples


def find_flares_with_overlap(df, overlap_percentage):
    df = df[df['total_overlap_percentage'] >= overlap_percentage]
    return df


def add_mwa_project_and_obs_ids(flares_df, mwa_data):
    flares_df['projectids'] = [[] for _ in range(len(flares_df))]
    flares_df['obs_ids'] = [[] for _ in range(len(flares_df))]
    
    for idx, row in flares_df.iterrows():
        flare_start, flare_end = pd.to_datetime(row['flare_start_UTC_corrected']), pd.to_datetime(row['flare_end_UTC_corrected'])
        matching_data = mwa_data[(mwa_data['starttime_utc'] <= flare_end) & (mwa_data['stoptime_utc'] >= flare_start)]
        flares_df.at[idx, 'projectids'] = matching_data['projectid'].tolist()
        flares_df.at[idx, 'obs_ids'] = matching_data['obs_id'].tolist()
        
    return flares_df


if __name__ == "__main__":
    main()
