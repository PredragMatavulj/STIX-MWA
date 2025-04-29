import os
import gc
import ast
import logging 
import traceback
import pandas as pd
from datetime import timedelta
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from helper_functions.spectrogram import get_spectrogram
from helper_functions.mwa import get_mwa_metadata, get_mwa_light_curve
from helper_functions.utils import get_root_path_to_data, safe_parse_time
from helper_functions.stix import get_flarelist, load_stix_light_curve, get_position


def main():
    """
    plots data based on provided observation ids or STIX flares matched with mwa metadata.
    """
    observations = ['1126847624']  # set to [] to use flare list

    if observations:
        save_folder = '../files/plots/spectrograms'
        plot_by_observations(observations, save_folder)
    else:
        for use_time_corrected in [True, False]:
            save_folder = (
                '../files/plots/spectrograms_and_light_curves/time_corrected'
                if use_time_corrected else
                '../files/plots/spectrograms_and_light_curves/time_not_corrected'
            )
            plot_by_flarelist(flare_range=None)  # or flare_range=(0, 3000)


def plot_by_observations(observations, save_folder):
    """
    plots spectrograms using manually specified observation IDs
    """
    os.makedirs(save_folder, exist_ok=True)

    save_name = 'spec_obs_' + '_'.join(observations)
    save_path = os.path.join(save_folder, save_name)

    plot_flare(save_path=save_path, obs_ids=observations)


def plot_by_flarelist(flare_range=None):
    """
    plots spectrograms and light curves using flare metadata
    """
    for use_time_corrected in [True, False]:
        save_folder = (
            '../files/plots/spectrograms_and_light_curves/time_corrected'
            if use_time_corrected else
            '../files/plots/spectrograms_and_light_curves/time_not_corrected'
        )
        os.makedirs(save_folder, exist_ok=True)

        flare_file = (
            "../info/flares_recorded_by_mwa_with_time_correction.csv"
            if use_time_corrected else
            "../info/flares_recorded_by_mwa_no_time_correction.csv"
        )
        flare_data = get_flarelist(flare_file)
        logging.info(f"Number of flares found: {len(flare_data)}")

        for i, flare_row in flare_data.iterrows():
            if flare_range and not (flare_range[0] <= i < flare_range[1]):
                continue

            logging.info(f"***** Processing flare {i} with ID {flare_row['flare_id']} *****")
            try:
                save_path = os.path.join(save_folder, f"{i+2}_flareID_{flare_row['flare_id']}")
                should_stop = plot_flare(save_path=save_path, row=flare_row)
                if should_stop:
                    logging.info(f"Continuing...")
                    continue
            except Exception as e:
                logging.error(f"{e} \n{traceback.format_exc()}")
            finally:
                plt.close('all')
                gc.collect()


def plot_flare(save_path, row=None, obs_ids=None):
    fig, axes, cbar_gs = create_figure_and_axes()
    try:
        if row is not None:
            spec, time_axis = plot_stix_and_mwa_from_row(row, axes, cbar_gs, fig, get_root_path_to_data())
        elif obs_ids is not None:
            spec, time_axis = plot_mwa_from_obs_ids(obs_ids, axes, cbar_gs, fig, get_root_path_to_data())
        else:
            raise ValueError("Either row or obs_ids must be provided.")

        if not time_axis:
            return True

        plot_mwa_light_curve(spec, time_axis, axes[2], row)
        if row is not None:
            plot_positions(time_axis, axes[3])
        else:
            axes[3].text(0.5, 0.5, 'STIX data not available!', ha='center', va='center')
            axes[3].axis('off')
        finalize_plot(fig, save_path)
    finally:
        plt.close(fig)

    return False


def create_figure_and_axes():
    fig = plt.figure(figsize=(10, 17))
    gs = GridSpec(4, 2, width_ratios=[1, 0.05], height_ratios=[1, 1, 1, 1])
    axes = [fig.add_subplot(gs[i, 0]) for i in range(4)]
    return fig, axes, gs


def plot_stix_and_mwa_from_row(row, axes, gs, fig, data_path):
    plot_stix_light_curve(row, axes[0], energy_range=(0, 4))
    return plot_mwa_spectrogram(row, axes[1], fig, gs, data_path)


def plot_mwa_from_obs_ids(obs_ids, axes, gs, fig, data_path):
    axes[0].text(0.5, 0.5, 'LC not available!', ha='center', va='center')
    axes[0].axis('off')

    mwa_metadata = get_mwa_metadata(obs_ids=obs_ids)
    spec, times, freqs = get_spectrogram(mwa_metadata, data_path)

    if spec is None:
        axes[1].text(0.5, 0.5, 'MWA spectrogram not available!', ha='center', va='center')
        return None, None

    im, time_axis = draw_mwa_spectrogram(spec, times, freqs, axes[1], safe_parse_time(times[0][0]), safe_parse_time(times[-1][-1]))
    cbar_ax = fig.add_subplot(gs[1, 1])
    plt.colorbar(im, cax=cbar_ax, label='Power')
    axes[1].set_title('Dynamic spectrum from MWA observations')

    return spec, time_axis


def plot_stix_light_curve(row, ax, energy_range):
    start_utc = row['stix_start_UTC']
    end_utc = row['stix_end_UTC']

    light_curve = load_stix_light_curve(start_utc, end_utc)

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
    set_x_ticks(ax)
    ax.set_yscale('log')
    ax.legend(loc='upper right')
    ax.set_title(f"STIX Light curves for the flare with ID {row['flare_id']}")
    ax.set_xlim(pd.to_datetime(start_utc), pd.to_datetime(end_utc))
    return ax


def set_x_ticks(ax):
    locator = AutoDateLocator(minticks=3, maxticks=7)
    formatter = ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)


def plot_mwa_spectrogram(flare_row, ax, fig, gs, path_to_data):
    start_time = flare_row["mwa_start_UTC"]
    end_time = flare_row["mwa_end_UTC"]

    mwa_metadata = get_mwa_metadata(start_time=start_time, end_time=end_time)

    if mwa_metadata.empty:
        ax.text(0.5, 0.5, 'MWA metadata not available!', ha='center', va='center')
        return ax, []
    
    spec, times, freqs = get_spectrogram(mwa_metadata, path_to_data)
    if spec is None or not times or not times[0]:
        return ax, []

    im, time_axis = draw_mwa_spectrogram(spec, times, freqs, ax, start_time, end_time)

    cbar_ax = fig.add_subplot(gs[1, 1])
    plt.colorbar(im, cax=cbar_ax)

    project_summary = get_project_summary(flare_row["projectids"])
    ax.set_title(f'Dynamic spectrum from MWA observations. \n project IDs: {project_summary}')
    return spec, time_axis


def draw_mwa_spectrogram(spec, times, freqs, ax, start_cut, end_cut, time_res=4):
    """
    draws mwa spectrogram using fixed 4s resolution and time-aligned x-axis, including gaps
    """
    start_time = safe_parse_time(times[0][0])
    end_time = safe_parse_time(times[0][1])
    num_cols = spec.shape[1]

     # generate equally spaced time axis between start_time and end_time
    dt = (end_time - start_time) / (num_cols - 1) if num_cols > 1 else timedelta(seconds=time_res)
    time_axis = [start_time + i * dt for i in range(num_cols)]

    im = ax.imshow(
        spec,
        aspect='auto',
        origin='lower',
        extent=[time_axis[0], time_axis[-1], freqs[0], freqs[-1]]
    )

    ax.set_ylabel('Frequency [MHz]')
    ax.set_xlabel('Time UTC')
     # restrict yticks to valid frequency range
    ax.set_yticks([yt for yt in ax.get_yticks() if freqs[0] <= yt <= freqs[-1]])
    set_x_ticks(ax)
     # set xlim to the start and end time of the flare
    ax.set_xlim(safe_parse_time(start_cut), safe_parse_time(end_cut))
    return im, time_axis


def get_project_summary(projectids):
    """
    get project summary from the metadata
    """
     # get the project ids from the metadata
    project_ids = ast.literal_eval(projectids)
     # count unique project ids
    project_counter = Counter(project_ids)
     # nicely format the result
    return ', '.join(f'{pid} ({count})' for pid, count in project_counter.items())


def plot_mwa_light_curve(spec, time_axis, ax, flare_row):
    """
    plots the integrated mwa light curve using a shared time axis
    """    
    light_curve = get_mwa_light_curve(spec)

    if light_curve is None:
        ax.text(0.5, 0.5, 'MWA light curve not available!', ha='center', va='center')
        return

    ax.plot(time_axis, light_curve)
    set_x_ticks(ax)

     # set xlim to the start and end time of the flare
    if flare_row is None:
        ax.set_xlim(time_axis[0], time_axis[-1])
    else:
        ax.set_xlim(safe_parse_time(flare_row["mwa_start_UTC"]), safe_parse_time(flare_row["mwa_end_UTC"]))
    ax.set_title("Integrated MWA light curve")
    ax.set_ylabel("Total intensity (log scale)")
    ax.set_xlabel("Time UTC")


def plot_positions(time_axis, ax):
    if len(time_axis) == 0 or time_axis is None:
        ax.text(0.5, 0.5, 'MWA spectrogram not available!', ha='center', va='center')
        return ax
    
    start = time_axis[0]
    end = time_axis[-1]

    emph = get_position(start, end)
    orbit = emph.data['orbit']

     # get solo position (first entry is enough since steps=1)
    x_solo = orbit['x'][0]
    y_solo = orbit['y'][0]

     # get earth position
    earth_x = orbit['objects']['earth']['x'][0]
    earth_y = orbit['objects']['earth']['y'][0]

    plot_object(ax, 0, 0, "Sun", "yellow", 1000)
    plot_object(ax, earth_x, earth_y, "Earth", "green", 250)
    plot_object(ax, x_solo, y_solo, "SOLO", "orange", 50)

     # formatting
    ax.set_title(f"SOLO Location at {orbit['utc'][0]}")
    ax.set_xlabel('X (au)')
    ax.set_ylabel('Y (au)')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.grid(True)


def plot_object(ax, x, y, label, color, size, marker='o', zorder=3):
    ax.scatter(x, y, s=size, c=color, label=label, edgecolors='black', marker=marker, zorder=zorder)
    ax.text(x, y, label, ha='center', va='center', zorder=zorder+1)


def finalize_plot(fig, save_path):
    # save the plot to a file and close the figure
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
