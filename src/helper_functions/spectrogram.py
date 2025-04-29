import os
import re
import shutil
import logging
import tarfile
import tempfile
import numpy as np
from casacore.tables import table, taql
from helper_functions.utils import safe_parse_time


def get_spectrogram(mwa_metadata, path_to_data):
    """
    processes all mwa observations in metadata and returns a merged spectrogram, time axis, and frequency axis
    """
    spectrograms = []
    times = []
    processed_ids = []
    for i, row in mwa_metadata.iterrows():
        result = process_single_observation(row, path_to_data)
        if result is None:
            continue
        dspec, time_range = result
        spectrograms.append(dspec)
        times.append(time_range)
        processed_ids.append(row['obs_id'])
        logging.info(f"Processed observation {row['obs_id']}")

    if not spectrograms:
        return None, None, None

    spec, time_axis, freq_axis = merge_spectrograms(spectrograms, times)
    return spec, time_axis, freq_axis


def process_single_observation(row, path_to_data):
    """
    returns (dspec, (start_time, end_time)) for one observation, or None on failure
    """
    obs_id = row['obs_id']
    data_file_path = get_raw_data_file_path(obs_id, path_to_data)
    if data_file_path is None:
        logging.info(f"No data for observation {obs_id}")
        return None

    try:
        dspec = get_dynamic_spec(fname=data_file_path, domedian=True)
        return dspec, (row['starttime_utc'], row['stoptime_utc'])
    except Exception as e:
        logging.info(f"Spectrogram error for obs {obs_id}: {e}")
        return None


def get_raw_data_file_path(obs_id, path_to_data):
    available_files = set(os.listdir(path_to_data))
    measurement = next((f for f in available_files if f.startswith(f"{obs_id}_")), None)
    return os.path.join(path_to_data, measurement) if measurement else None


def get_dynamic_spec(fname=None, domedian=True):
    try:
        ms_files, temp_dir = get_ms_files(fname)
    except Exception as e:
        logging.info(f"Error extracting MS files: {e}")
    spectrograms, frequencies = [], []

    try:
        for ms_file in ms_files:
            ms_path = os.path.join(temp_dir, ms_file.name)
            ms_path = os.path.normpath(ms_path)

             # query the entire DATA column as a 3D array (nfreq × npol)
            result = taql(f"SELECT DATA FROM '{ms_path}'")
            data = result.getcol("DATA")  # shape: (nrows, nchan, npol)

             # convert complex data to amplitude and extract polarization
            amp = ( np.abs(data[:, :, 0]) + np.abs(data[:, :, 0]) ) / 2
            amp = np.abs(data[:, :, 0])  # shape: (nrows, nchan)
            amp = amp.T  # shape: (nchan, nrows) -> (freq, time)

             # get number of baselines
            nbl = get_nbl(ms_path)
            ntime = data.shape[0] // nbl

            freqs = get_frequencies(ms_path)
            frequencies.extend([int(np.round(f)) for f in freqs])

             # reshape and process
            amp = ( np.abs(data[:, :, 0]) + np.abs(data[:, :, 3])) / 2  # average polarizations 0 and 3
            amp = amp.reshape((ntime, nbl, -1))  # (time, baseline, freq)

             # mask very small and invalid values
            amp = np.ma.masked_where(amp < 1e-9, amp)
            amp = np.ma.masked_invalid(amp)

             # average over baselines
            if domedian:
                ospec = np.ma.median(amp, axis=1).T  # shape: (freq, time)
            else:
                ospec = np.ma.mean(amp, axis=1).T

             # log scale specrogram
            ospec = np.ma.masked_invalid(ospec)
            ospec = np.ma.log2(np.ma.clip(ospec, 1, None))

            spectrograms.append(ospec)

    finally:
         # clean up the temporary directory after your operations
        shutil.rmtree(temp_dir)

    dspec_entity = {
        "spec": np.concatenate(spectrograms, axis=0),
        "freq": frequencies
    }
    return dspec_entity


def get_ms_files(fname):
    """
    extracts ms files from a tar archive and returns them sorted by channel number if available.
    """
    temp_dir = tempfile.mkdtemp()
    with tarfile.open(fname, 'r') as tar:
        ms_files = [member for member in tar.getmembers() if member.name.endswith('.ms')]
        if not ms_files:
            raise ValueError("No .ms files found in the tar archive.")
        else:
            tar.extractall(path=temp_dir)
     # sort by the channel number extracted from the filename
    ms_files = sorted(ms_files, key=lambda f: _extract_channel_or_fallback(f.name))
    return ms_files, temp_dir


def _extract_channel_or_fallback(name):
    """
    extracts channel number from filename or falls back to a large number to push it to the end
    """
    match = re.search(r'ch(\d+)(?:-|\.ms)', name)
    if match:
        return int(match.group(1))
    else:
         # fallback: extract the full number if filename is like '1355089520.ms'
        match = re.search(r'(\d+)', name)
        if match:
            return int(match.group(1))
        else:
            return float('inf')  # put files without any number last


def get_nbl(ms_path):
    """
    get the number of antennas and baselines in a measurement set.
    """
    ant_table = table(f"{ms_path}/ANTENNA", ack=False)
    nant = len(ant_table)
    ant_table.close()

    nbl = nant * (nant - 1) // 2 # number of unique baselines (cross-correlations only)
    nbl += nant # include autocorrelations
    return nbl


def get_frequencies(ms_path):
    """
    get frequences in a measurement set.
    """
    spw = table(f"{ms_path}/SPECTRAL_WINDOW", ack=False)
    chan_freq = spw.getcol("CHAN_FREQ")[0]  # shape: (nchan,)
    spw.close()

    freqs_mhz = chan_freq / 1e6  # convert Hz to MHz
    return freqs_mhz


def merge_spectrograms(spectrograms, times, time_res=4):
    """
    merges spectrograms with time-aligned gaps, rounding timestamps to seconds
    """
    freq_axis = None
    combined_spec, combined_times = [], []

    for i, (dspec, (start, end)) in enumerate(zip(spectrograms, times)):
        start = safe_parse_time(start)
        end = safe_parse_time(end)
        spec = dspec['spec']
        freq = dspec['freq']

        if freq_axis is None:
            freq_axis = freq
        elif not np.array_equal(freq_axis, freq):
            logging.info(f"frequency mismatch in observation {i}, skipping...")
            continue

        if i > 0:
            prev_end = safe_parse_time(times[i - 1][1])
            if start > prev_end:
                gap_sec = (start - prev_end).total_seconds()
                gap_cols = int(np.round(gap_sec / time_res))
                nan_gap = np.full((spec.shape[0], gap_cols), np.nan)
                combined_spec.append(nan_gap)
                combined_times.append((prev_end, start))

        combined_spec.append(spec)
        combined_times.append((start, end))

    final_spec = np.hstack(combined_spec) if combined_spec else None
    return final_spec, combined_times, freq_axis
