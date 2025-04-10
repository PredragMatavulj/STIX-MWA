import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from dotenv import load_dotenv
from stixdcpy.quicklook import LightCurves
from helper_functions.dspec import get_dspec
import rootutils
from pathlib import Path
import pyvo
from datetime import datetime, timezone
from dateutil import parser
from matplotlib.gridspec import GridSpec


def main():
    setup_environment()
    flare_data = pd.read_csv('../info/flares_recorded_by_mwa.csv')
    save_folder = '../files/plots/spectrograms_and_light_curves'

    for i, flare_row in flare_data.iterrows():
        try:
            save_path = os.path.join(save_folder, f"{i}_flare_{flare_row['flare_id']}")
            plot_flare(flare_row, save_path)
        except Exception as e:
            print(f"Error in processing flare {i}: {e}")


def setup_environment():
    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
    load_dotenv()
    global ROOT_PATH_TO_DATA
    ROOT_PATH_TO_DATA = Path(os.getenv("ROOT_PATH_TO_DATA", "default/path/to/data"))


def plot_flare(row, save_path):
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 2, width_ratios=[1, 0.05], height_ratios=[1, 1])

     # Use the first column for the actual plots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0]) 
    plot_light_curve(row, ax1, energy_range=(0, 4))
    plot_spectrogram(row, ax2, fig, gs)
    finalize_plot(fig, save_path)


def plot_light_curve(row, ax, energy_range):
    light_curve = load_light_curve(row['flare_start_UTC'], row['flare_end_UTC'])

    if not light_curve.data and not light_curve:
        ax.text(0.5, 0.5, 'LC not available!', ha='center', va='center')
        return ax

    for i in range(5) if energy_range is None else range(energy_range[0], energy_range[1] + 1):
        ax.plot(
            light_curve.time,
            light_curve.counts[i, :],
            label=light_curve.energy_bins['names'][i]
        )
    ax.set_ylabel('Counts')
    locator = AutoDateLocator(minticks=3, maxticks=7)
    formatter = ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_yscale('log')
    ax.legend(loc='upper right')
    ax.set_title("Light curves of STIX data for the whole flare")
    ax.set_xlim(pd.to_datetime(row['flare_start_UTC']), pd.to_datetime(row['flare_end_UTC']))
    return ax


def load_light_curve(start_utc, end_utc):
    try:
        return LightCurves.from_sdc(start_utc, end_utc, ltc=True)
    except Exception as e:
        print(f"Error loading light curves: {e}")
        return None


def plot_spectrogram(row, ax, fig, gs):
    start_time = row["flare_start_UTC_corrected"]
    end_time = row["flare_end_UTC_corrected"]
    mwa_metadata = get_mwa_metadata(start_time, end_time)

    spec, times, freqs = get_spectrogram(mwa_metadata)
    if spec is None:
        ax.text(0.5, 0.5, 'MWA spectrogram not available!', ha='center', va='center')
        return ax

     # convert time strings to datetime objects, handling ISO format with timezone
    times = [(parser.parse(start), parser.parse(end)) for start, end in times]

     # prepare the time axis with gaps represented as breaks (using NaNs) (if there are any missing observations)
    time_axis = []
    num_columns = spec.shape[1]
    columns_per_segment = num_columns // len(times)  # calculate columns per segment roughly
    current_column = 0

    for start, end in times:
        if current_column > 0:  # if not the first segment
             # add a NaN gap before starting new segment
            time_axis.extend([np.nan] * columns_per_segment)
         # create a time range for the current segment
        segment_time = np.linspace(start.timestamp(), end.timestamp(), columns_per_segment)
        time_axis.extend(segment_time)
        current_column += columns_per_segment

    time_axis = np.array(time_axis)
    valid_times = ~np.isnan(time_axis)
    time_axis = [datetime.fromtimestamp(t, timezone.utc) for t in time_axis[valid_times]]

    im = ax.imshow(spec, aspect='auto', origin='lower', extent=[min(time_axis), max(time_axis), freqs[0], freqs[-1]])
    ax.set_ylabel('Frequency [MHz]')
    ax.set_xlabel('Time UTC')
    
     # adding a colorbar only next to the first subplot
    cbar_ax = fig.add_subplot(gs[1, 1])  # Colorbar position
    plt.colorbar(im, cax=cbar_ax, label='Power')

    # Adjust y-axis tick labels to display frequencies in MHz
    yticks = ax.get_yticks()
    valid_yticks = [ytick for ytick in yticks if freqs[0] <= ytick <= freqs[-1]]  # Filter ticks within frequency range
    ax.set_yticks(valid_yticks)  # Explicitly set the valid tick positions
    ax.set_yticklabels([f"{ytick / 1e6:.0f}" for ytick in valid_yticks])  # Convert to MHz and format

    # Set xticks at the beginning, end, and two in the middle
    xticks = [time_axis[0], time_axis[len(time_axis) // 3], time_axis[2 * len(time_axis) // 3], time_axis[-1]]
    ax.set_xticks(xticks)
    ax.set_xticklabels([t.strftime('%H:%M:%S') for t in xticks])  # Format x-tick labels

    # Add the date next to the x-axis ticks, right bottom before x label
    date_str = time_axis[0].strftime('%Y-%m-%d')
    ax.annotate(date_str, xy=(1, -0.1), xycoords='axes fraction', ha='right', fontsize=10)

    ax.set_title('Dynamic Spectrum from MWA observations')


def get_mwa_metadata(start_time, end_time):
     
    tap_service = pyvo.dal.TAPService("http://vo.mwatelescope.org/mwa_asvo/tap")
    query = f"""
    SELECT * FROM mwa.observation
    WHERE stoptime_utc >= '{format_time(start_time)}' 
    AND starttime_utc <= '{format_time(end_time)}'
    """

    result = tap_service.search(query)
    mwa_metadata = result.to_table().to_pandas()    # contains all observations for a specific flare
    print(f"Number of found observations is {len(mwa_metadata)}")

    if mwa_metadata.empty:
        raise ValueError("No MWA observations found for the specified time range.")
    
    return mwa_metadata


def format_time(time_str):
    dt = parser.parse(time_str)
     # format the datetime to the desired format, cutting off milliseconds to 3 digits
    formatted_time = dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    return formatted_time


def get_spectrogram(mwa_metadata):

    spectrograms, processed, unprocessed, times = [], [], [], []
    last_processed_time = None

    # process each observation
    for _, row in mwa_metadata.iterrows():
        obs_id = row['obs_id']
        start_time, end_time = row['starttime_utc'], row['stoptime_utc']
        dspec = None

        data_file_path = get_raw_data_file_path(obs_id)
        if data_file_path is None:
            unprocessed.append(obs_id)
            print(f" \n No data found for observation {obs_id}.")
            continue

        try:
            dspec = get_dspec(fname=data_file_path, domedian=True)
        except Exception as e:
            unprocessed.append(obs_id)
            print(f" \n Error creating spectrogram for observation {obs_id}: {e}")
            continue

        spectrograms.append(dspec)
        processed.append(obs_id)
        times.append((last_processed_time or start_time, end_time))
        last_processed_time = end_time
        print(f" \n Successfully processed observation {obs_id}.")

    # merge spectrograms and plot if any were successfully created
    spec, time, freq = None, None, None
    if spectrograms:
        try:
            spec, time, freq = merge_spectrograms(spectrograms, times)
        except Exception as e:
            print(f" \n Error merging or plotting spectrograms: {e}")
    else:
        print(" \n No spectrograms were successfully created.")

    return spec, time, freq


def get_raw_data_file_path(obs_id):
    available_files = set(os.listdir(ROOT_PATH_TO_DATA))
    measurement = next((f for f in available_files if f.startswith(f"{obs_id}_")), None)
    if measurement is None:
        file_path = None
    else:
        file_path = os.path.join(ROOT_PATH_TO_DATA, measurement)
    return file_path


def merge_spectrograms(spectrograms, times):
    """
    merges multiple spectrogram data into a single combined spectrogram
    """
     # initialize variables
    freq_axis = None
    combined_spec = None
    combined_times = []
    last_end_time = None

    for dspec, time in zip(spectrograms, times):
        spec = dspec['spec']
        freq = dspec['freq']
        start_time, end_time = time

         # if there is a gap between the current and previous time intervals, add it to combined_times
        if last_end_time is not None and start_time > last_end_time:
            combined_times.append((last_end_time, start_time))

         # initialize the frequency axis if not already set
        if freq_axis is None:
            freq_axis = freq
        # check for consistency in frequency axes across spectrograms
        elif not np.array_equal(freq_axis, freq):
            raise ValueError("Frequency axes do not match across measurements!")

         # initialize or append the spectrogram data
        if combined_spec is None:
            combined_spec = spec
        else:
            combined_spec = np.hstack((combined_spec, spec))  # concatenate spectrograms horizontally

         # add the current time interval to combined_times
        combined_times.append((start_time, end_time))
        last_end_time = end_time  # update the last end time

    return combined_spec, combined_times, freq_axis


def finalize_plot(fig, save_path):
    # save the plot to a file and close the figure
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
