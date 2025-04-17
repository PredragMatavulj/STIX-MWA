import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from dotenv import load_dotenv
from stixdcpy.quicklook import LightCurves
from helper_functions.dspec import get_dspec2 as get_dspec
import rootutils
from pathlib import Path
import pyvo
from datetime import datetime, timezone
from dateutil import parser
from matplotlib.gridspec import GridSpec
import find_flares_in_mwa


def main():
    """
    Plot light curves and spectrograms for flares. 
    If observations contain a list of observations, it will plot the combined spectrogram for those observations.
    If not, it will plot the light curves and spectrograms for the flares listed in the CSV file.
    """
    setup_environment()
    info_path = '../info'

    flares_in_mwa = 'flares_recorded_by_mwa.csv'
     # if the file is not found, run the find_flares_in_mwa script to create it
    if flares_in_mwa not in os.listdir(info_path):
        find_flares_in_mwa.main()
    flare_data = pd.read_csv(os.path.join(info_path, flares_in_mwa))
    
    observations = ['1126847624']

    if observations:
        save_folder = '../files/plots/spectrograms'

        save_name = 'spec_obs'
        for obs_id in observations:
            save_name += f"_{obs_id}"

        save_path = os.path.join(save_folder, save_name)
        plot_flare(save_path=save_path, obs_ids=observations)
    else:
        save_folder = '../files/plots/spectrograms_and_light_curves'

        for i, flare_row in flare_data.iterrows():
            try:
                save_path = os.path.join(save_folder, f"{i}_flare_{flare_row['flare_id']}")
                plot_flare(save_path=save_path, row=flare_row)
            except Exception as e:
                print(f"\n Error in processing flare {i}: {e}")


def setup_environment():
    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
    load_dotenv()
    global ROOT_PATH_TO_DATA
    ROOT_PATH_TO_DATA = Path(os.getenv("ROOT_PATH_TO_DATA", "default/path/to/data"))


def plot_flare(save_path, row=None, obs_ids=None):
    fig = plt.figure(figsize=(10, 12))
    gs = GridSpec(3, 2, width_ratios=[1, 0.05], height_ratios=[1, 1, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])

    if row is not None:
        plot_stix_light_curve(row, ax1, energy_range=(0, 4))
        spec, time_axis = plot_mwa_spectrogram(row, ax2, fig, gs)
    elif obs_ids is not None:
        mwa_metadata = get_mwa_metadata(obs_ids=obs_ids)
        spec, times, freqs = get_spectrogram(mwa_metadata)
        if spec is None:
            ax2.text(0.5, 0.5, 'MWA spectrogram not available!', ha='center', va='center')
        else:
            im, time_axis = draw_mwa_spectrogram(spec, times, freqs, ax2)
            cbar_ax = fig.add_subplot(gs[1, 1])
            plt.colorbar(im, cax=cbar_ax, label='Power')
            ax2.set_title('Dynamic Spectrum from MWA observations')
        ax1.axis('off')  # hide the top subplot since there's no light curve

    plot_mwa_light_curve(spec, time_axis, ax=ax3)
    finalize_plot(fig, save_path)


def plot_stix_light_curve(row, ax, energy_range):
    light_curve = load_stix_light_curve(row['flare_start_UTC'], row['flare_end_UTC'])

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


def load_stix_light_curve(start_utc, end_utc):
    try:
        return LightCurves.from_sdc(start_utc, end_utc, ltc=True)
    except Exception as e:
        print(f"\n Error loading light curves: {e}")
        return None


def plot_mwa_spectrogram(row, ax, fig, gs):
    start_time = row["flare_start_UTC_corrected"]
    end_time = row["flare_end_UTC_corrected"]
    mwa_metadata = get_mwa_metadata(start_time=start_time, end_time=end_time)

    spec, times, freqs = get_spectrogram(mwa_metadata)
    if spec is None:
        ax.text(0.5, 0.5, 'MWA spectrogram not available!', ha='center', va='center')
        return ax

    im, time_axis = draw_mwa_spectrogram(spec, times, freqs, ax)

    cbar_ax = fig.add_subplot(gs[1, 1])
    plt.colorbar(im, cax=cbar_ax)
    ax.set_title('Dynamic Spectrum from MWA observations')
    return spec, time_axis


def draw_mwa_spectrogram(spec, times, freqs, ax=None):
     # parse times
    times = [(parser.parse(start), parser.parse(end)) for start, end in times]

     # prepare time axis
    time_axis = []
    num_columns = spec.shape[1]
    columns_per_segment = num_columns // len(times)
    current_column = 0

    for start, end in times:
        if current_column > 0:
            time_axis.extend([np.nan] * columns_per_segment)
        segment_time = np.linspace(start.timestamp(), end.timestamp(), columns_per_segment)
        time_axis.extend(segment_time)
        current_column += columns_per_segment

    time_axis = np.array(time_axis)
    valid_times = ~np.isnan(time_axis)
    time_axis = [datetime.fromtimestamp(t, timezone.utc) for t in time_axis[valid_times]]

    if ax is None:
        ax = plt.gca()

    im = ax.imshow(spec, aspect='auto', origin='lower', extent=[min(time_axis), max(time_axis), freqs[0], freqs[-1]])
    ax.set_ylabel('Frequency [MHz]')
    ax.set_xlabel('Time UTC')

    yticks = ax.get_yticks()
    valid_yticks = [ytick for ytick in yticks if freqs[0] <= ytick <= freqs[-1]]
    ax.set_yticks(valid_yticks)

    xticks = [time_axis[0], time_axis[len(time_axis) // 3], time_axis[2 * len(time_axis) // 3], time_axis[-1]]
    ax.set_xticks(xticks)
    ax.set_xticklabels([t.strftime('%H:%M:%S') for t in xticks])

    date_str = time_axis[0].strftime('%Y-%m-%d')
    ax.annotate(date_str, xy=(1, -0.1), xycoords='axes fraction', ha='right', fontsize=10)

    return im, time_axis


def get_mwa_metadata(start_time=None, end_time=None, obs_ids=None):
    tap_service = pyvo.dal.TAPService("http://vo.mwatelescope.org/mwa_asvo/tap")

     # define the query based on the input parameters
    if obs_ids is not None:
        ids_formatted = ', '.join(f"'{id}'" for id in obs_ids)
        query = f"SELECT * FROM mwa.observation WHERE obs_id IN ({ids_formatted})"
    elif start_time is not None and end_time is not None:
        query = f"""
        SELECT * FROM mwa.observation
        WHERE stoptime_utc >= '{format_time(start_time)}'
        AND starttime_utc <= '{format_time(end_time)}'
        """
    else:
        raise ValueError("Invalid parameters. Provide either 'obs_id' or both 'start_time' and 'end_time'.")

     # execute the query
    result = tap_service.search(query)
    mwa_metadata = result.to_table().to_pandas()  # converts the result to a pandas DataFrame

    print(f"\n Number of found observations is {len(mwa_metadata)}")    
    if mwa_metadata.empty:
        raise ValueError("No MWA observations found for the specified criteria.")

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
            print(f"\n No data found for observation {obs_id}.")
            continue

        try:
            dspec = get_dspec(fname=data_file_path, domedian=True)
        except Exception as e:
            unprocessed.append(obs_id)
            print(f"\n Error creating spectrogram for observation {obs_id}: {e}")
            continue

        spectrograms.append(dspec)
        processed.append(obs_id)
        times.append((last_processed_time or start_time, end_time))
        last_processed_time = end_time
        print(f"\n Successfully processed observation {obs_id}.")

    # merge spectrograms and plot if any were successfully created
    spec, time, freq = None, None, None
    if spectrograms:
        try:
            spec, time, freq = merge_spectrograms(spectrograms, times)
        except Exception as e:
            print(f"\n Error merging or plotting spectrograms: {e}")
    else:
        print("\n No spectrograms were successfully created.")

    return spec, time, freq


def plot_mwa_light_curve(spec, time_axis, ax=None):
    """
    plots the integrated mwa light curve using a shared time axis
    """
    light_curve = np.ma.sum(spec, axis=0)  # shape: (time,)

    if ax is None:
        ax = plt.gca()

    ax.plot(time_axis, light_curve)
    ax.set_xlim(time_axis[0], time_axis[-1])
    ax.set_title("Integrated MWA light curve")
    ax.set_ylabel("Total intensity (log scale)")
    ax.set_xlabel("Time UTC")

    locator = AutoDateLocator(minticks=3, maxticks=7)
    formatter = ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    return ax


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
