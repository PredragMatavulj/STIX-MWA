import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from dotenv import load_dotenv, dotenv_values
from stixdcpy.quicklook import LightCurves
from mwa_spectrograms_raw_codes.dspec import get_dspec
import rootutils
import logging
from pathlib import Path
from memory_profiler import profile, memory_usage

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Profile memory usage for specific functions
get_dspec = profile(get_dspec)


def main():
    setup_environment()
    flare_data, mwa_metadata = load_data()
    spectrogram_folder = '../files/plots/spectrograms_and_light_curves'
    for i, flare_row in flare_data.iterrows():
        if i+2 != 3:
            continue
        plot_flare(flare_row, mwa_metadata, spectrogram_folder)
    plot_memory_usage(flare_data, mwa_metadata, spectrogram_folder)


def setup_environment():
    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
    load_dotenv()
    os.environ.update({**os.environ, **dotenv_values(".env")})
    global ROOT_PATH_TO_DATA
    ROOT_PATH_TO_DATA = Path(os.getenv("ROOT_PATH_TO_DATA", "default/path/to/data"))
    assert ROOT_PATH_TO_DATA.exists(), "Data path does not exist."


def load_data():
    flare_data = pd.read_csv('../data/flares_recorded_by_mwa.csv')
    mwa_metadata = pd.read_csv('../data/mwa_metadata.csv', low_memory=False)
    return flare_data, mwa_metadata


def plot_flare(row, mwa_metadata, spectrogram_folder, index):
    flare_id = row["flare_id"]
    obs_ids = safe_eval(row['obs_ids'])

    mwa_observations = filter_observations(mwa_metadata, obs_ids)
    mwa_starttime, mwa_endtime = combine_observation_times(mwa_observations)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    light_curve = load_light_curve(row['flare_start_UTC'], row['flare_end_UTC'])
    if light_curve and light_curve.data:
        plot_light_curve(light_curve, ax1, energy_range=(0, 4))

    ax1.set_xlim(pd.to_datetime(row['flare_start_UTC']), pd.to_datetime(row['flare_end_UTC']))

    process_and_plot_spectrogram(obs_ids, spectrogram_folder, ax2, mwa_starttime, mwa_endtime)

    finalize_plot(fig, ax2, spectrogram_folder, index, flare_id)


def safe_eval(obs_ids_str):
    # safely evaluate string of observation IDs
    try:
        return eval(obs_ids_str, {"__builtins__": None}, {})
    except SyntaxError as e:
        log.error(f"Failed to parse observation IDs: {e}")
        return []


def filter_observations(metadata, obs_ids):
    # filter metadata for observations
    observations = metadata[metadata['obs_id'].isin(obs_ids)].copy()
    observations['starttime_utc'] = pd.to_datetime(observations['starttime_utc'], errors='coerce')
    observations['stoptime_utc'] = pd.to_datetime(observations['stoptime_utc'], errors='coerce')
    return observations


def load_light_curve(start_utc, end_utc):
    try:
        return LightCurves.from_sdc(start_utc, end_utc, ltc=True)
    except Exception as e:
        log.error(f"Error loading light curves: {e}")
        return None


def plot_light_curve(curve, ax, energy_range):
    if not curve.data:
        ax.text(0.5, 0.5, 'LC not available!', ha='center', va='center')
        return ax

    for i in range(5) if energy_range is None else range(energy_range[0], energy_range[1] + 1):
        ax.plot(
            curve.time,
            curve.counts[i, :],
            label=curve.energy_bins['names'][i]
        )
    ax.set_ylabel('Counts')
    locator = AutoDateLocator(minticks=3, maxticks=7)
    formatter = ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_yscale('log')
    ax.legend(loc='upper right')
    ax.set_title("Light curves of STIX data for the whole flare")
    return ax


def process_and_plot_spectrogram(obs_ids, folder, ax, start_time, end_time):
    # process observations and plot spectrogram
    try:
        existing_files, missing_files, combined_spec, freq_axis = process_flare(obs_ids, folder)
        if existing_files:
            ax.imshow(combined_spec, aspect='auto', origin='lower', extent=[start_time, end_time, freq_axis.min(), freq_axis.max()])
            ax.set_ylabel('Frequency [MHz]')
            ax.set_xlabel('Time UTC')
        else:
            ax.text(0.5, 0.5, 'MWA data not available!', ha='center', va='center')
    except MemoryError:
        log.warning(f"Skipping due to memory error.")


def finalize_plot(fig, ax, folder, index, flare_id):
    # save the plot to a file and close the figure
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"{index}_flare_{flare_id}.png"))
    plt.close(fig)


def combine_observation_times(observation_times):
    """Combine observation times into a single continuous or segmented timeline."""
    try:
        observation_times.sort_values(by='starttime_utc', inplace=True)
        gaps = observation_times['starttime_utc'].iloc[1:].values - observation_times['stoptime_utc'].iloc[:-1].values
        if (gaps > pd.Timedelta(seconds=5)).any():
            raise ValueError("Significant gaps detected between observations.")
        return observation_times.iloc[0]['starttime_utc'], observation_times.iloc[-1]['stoptime_utc']
    except Exception as e:
        log.info(f"Error processing observation times: {e}")


def process_flare(obs_ids, spectrogram_folder):
    existing_files = []
    missing_files = []
    freq_axis = None
    combined_spec = None

    # cache directory contents to reduce file system access
    available_files = set(os.listdir(ROOT_PATH_TO_DATA))

    for obs_id in obs_ids:
        # find matching files efficiently using set operations
        matching_files = [f for f in available_files if f.startswith(f"{obs_id}_")]

        if matching_files:
            try:
                dspec = create_spectrogram(matching_files[0], ROOT_PATH_TO_DATA, spectrogram_folder)
                log.info(f"Successfully processed observation {obs_id}.")
            except MemoryError:
                log.warning(f"Memory error encountered when processing observation {obs_id}. Skipping...")
                continue
            except Exception as e:
                log.error(f"Error processing observation {obs_id}: {e}")
                continue

            spec = dspec['spec']
            freq = dspec['freq']

            # initialize or validate the frequency axis
            if freq_axis is None:
                freq_axis = freq
            elif not np.array_equal(freq_axis, freq):
                log.error("Frequency axes do not match across measurements!")
                raise ValueError("Frequency axes do not match across measurements!")

            # initialize or concatenate spectrogram data
            if combined_spec is None:
                combined_spec = spec
            else:
                combined_spec = np.hstack((combined_spec, spec))

            existing_files.extend(matching_files)
        else:
            missing_files.append(obs_id)
            log.info(f"No matching files found for observation {obs_id}")

    return existing_files, missing_files, combined_spec, freq_axis


def create_spectrogram(measurement, root_path, spectrogram_folder):
    file_path = os.path.join(root_path, measurement)
    output_specfile = os.path.join(spectrogram_folder, f"{measurement}.npz")
    ensure_directory(spectrogram_folder)
    return get_dspec(fname=file_path, specfile=output_specfile, verbose=True)


def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def plot_memory_usage(flare_data, mwa_metadata, spectrogram_folder):
    mem_usage = memory_usage((plot_flare, (flare_data, mwa_metadata, spectrogram_folder)), interval=0.1)
    plt.figure(figsize=(10, 6))
    plt.plot(mem_usage, label='Memory Usage (MiB)')
    plt.title('Memory Usage Over Time')
    plt.xlabel('Time (0.1s intervals)')
    plt.ylabel('Memory Usage (MiB)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("../files/plots/memory_usage.png")

if __name__ == "__main__":
    main()
