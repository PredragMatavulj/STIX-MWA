import os
import time
import ast
import pandas as pd
import rootutils
from pathlib import Path
from dotenv import load_dotenv
import find_flares_in_mwa
from helper_functions.utils_mwa_asvo import initialize_settings, initialize_queues_and_locks, login_and_submit_jobs, start_status_thread, initialize_notifier, start_download_threads, handle_results, cleanup


def main():
    """
    Find and download MWA data.
    If no observations are provided, it will find and download MWA data corresponding to flares obtained by STIX data.
    """
    setup_environment()

    time_resolution = 60
    freq_resolution = 240
    observations = ['1126847624']  # Example observation ID

    if observations:
        download_mwa_data_based_on_observations(observations, time_resolution, freq_resolution)
    else:
        info_path = '../info'
        flares_in_mwa = 'flares_recorded_by_mwa.csv'
         # if the file is not found, run the find_flares_in_mwa script to create it
        if flares_in_mwa not in os.listdir(info_path):
            find_flares_in_mwa.main()

        flare_data = pd.read_csv(os.path.join(info_path, flares_in_mwa))
        downloaded_ids = get_downloaded_obs_ids(ROOT_PATH_TO_DATA)
        for i, row in flare_data.iterrows():
            download_mwa_data_based_on_flare_list(row, downloaded_ids)
            break


def setup_environment():
    """Sets up the necessary environment for the project."""
     # setup project root and load environment variables
    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
    load_dotenv()
    global ROOT_PATH_TO_DATA
    ROOT_PATH_TO_DATA = Path(os.getenv("ROOT_PATH_TO_DATA", "default/path/to/data"))


def download_mwa_data_based_on_observations(observations, avg_time_res, avg_freq_res):
    """
    prepares a list of jobs for given observation id and processes them to download mwa data
    """
    jobs_to_submit = create_jobs(observations, avg_time_res, avg_freq_res)
    process_mwa_asvo_jobs(jobs_to_submit)


def download_mwa_data_based_on_flare_list(row, downloaded_ids):
    print(f"Processing flare {row['flare_id']}.")
    start = time.time()

    obs_ids = ast.literal_eval(row['obs_ids'])
    new_obs_ids = [obs_id for obs_id in obs_ids if obs_id not in downloaded_ids]

    if new_obs_ids:
        jobs_to_submit = create_jobs(new_obs_ids)
        if jobs_to_submit:
            process_mwa_asvo_jobs(jobs_to_submit)

    end = time.time()
    print(f"All tasks completed for flare {row['flare_id']} in {(end - start) / 60:.2f} minutes!")


def get_downloaded_obs_ids(root_path):
    downloads = os.listdir(root_path)
    return {int(d.split("_")[0]) for d in downloads}


def create_jobs(observations, time_resolution, freq_resolution):
    """
    creates a list of job specifications for the mwa asvo jobs based on observation ids
    """
    return [
        (
            'submit_conversion_job_direct',
            {
                'obs_id': obs_id,
                'job_type': 'c',
                'avg_time_res': time_resolution,
                'avg_freq_res': freq_resolution,
                'output': 'ms',
            }
        )
        for obs_id in observations
    ]


def process_mwa_asvo_jobs(jobs):
    """
    processes a list of jobs by initializing the settings, submitting, and downloading the results
    """
    params, sslopt, verbose = initialize_settings()
    submit_lock, download_queue, result_queue, status_queue = initialize_queues_and_locks()
    session, jobs_list = login_and_submit_jobs(params, download_queue, status_queue, jobs)
    start_status_thread(status_queue)
    notify = initialize_notifier(params, sslopt, submit_lock, jobs_list, download_queue, result_queue, status_queue, verbose)
    threads = start_download_threads(submit_lock, jobs_list, download_queue, result_queue, status_queue, session, ROOT_PATH_TO_DATA)
    results = handle_results(submit_lock, jobs_list, result_queue, download_queue, threads)
    cleanup(notify, threads, result_queue, status_queue, results)
 

if __name__ == "__main__":
    main()
