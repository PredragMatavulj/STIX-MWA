import os
import ssl
import time
import ast
import pandas as pd
from mantaray.scripts.mwa_client import submit_jobs, status_func, download_func, notify_func
from mantaray.api import Session, Notify
from queue import Queue, Empty
from threading import Thread, RLock
import rootutils
from pathlib import Path
from dotenv import dotenv_values, load_dotenv

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #
ENV_VARIABLES = {**os.environ, **dotenv_values(".env")}
os.environ.update(ENV_VARIABLES)

ROOT_PATH_TO_DATA = Path(ENV_VARIABLES["ROOT_PATH_TO_DATA"])
ROOT_SERVER_URL = ENV_VARIABLES["SERVER_URL"]
ROOT_SERVER_PORT = ENV_VARIABLES["SERVER_PORT"]
ROOT_SERVER_HTTPS = ENV_VARIABLES["SERVER_HTTPS"]
ROOT_MWA_API_KEY = ENV_VARIABLES["MWA_API_KEY"]


def main():
    setup_environment()
    flare_data = load_flare_data('../data/flares_recorded_by_mwa.csv')
    downloaded_ids = get_downloaded_obs_ids(ROOT_PATH_TO_DATA)
    for i, row in flare_data.iterrows():
        submit_and_download_jobs_for_new_flare_observations(row, downloaded_ids)


def setup_environment():
    """Sets up the necessary environment for the project."""
    # Setup project root and environment variables
    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
    # ------------------------------------------------------------------------------------ #
    # the setup_root above is equivalent to:
    # - adding project root dir to PYTHONPATH
    #       (so you don't need to force user to install project as a package)
    #       (necessary before importing any local modules e.g. `from src import utils`)
    # - setting up PROJECT_ROOT environment variable
    #       (which is used as a base for paths in "configs/paths/default.yaml")
    #       (this way all filepaths are the same no matter where you run the code)
    # - loading environment variables from ".env" in root dir
    #
    # you can remove it if you:
    # 1. either install project as a package or move entry files to project root dir
    # 2. set `root_dir` to "." in "configs/paths/default.yaml"
    #
    # more info: https://github.com/ashleve/rootutils
    # ------------------------------------------------------------------------------------ #
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Update environment with loaded variables
    os.environ.update({**os.environ, **dotenv_values(".env")})
    
    # Validate and set global paths and settings
    global ROOT_PATH_TO_DATA, ROOT_SERVER_URL, ROOT_SERVER_PORT, ROOT_SERVER_HTTPS, ROOT_MWA_API_KEY
    ROOT_PATH_TO_DATA = Path(os.getenv("ROOT_PATH_TO_DATA", "default/path/to/data"))
    ROOT_SERVER_URL = os.getenv("SERVER_URL", "asvo.mwatelescope.org")
    ROOT_SERVER_PORT = os.getenv("SERVER_PORT", "443")
    ROOT_SERVER_HTTPS = bool(int(os.getenv("SERVER_HTTPS", "1")))
    ROOT_MWA_API_KEY = os.getenv("MWA_API_KEY", "default_api_key")

    validate_settings()


def validate_settings():
    # validates critical environment settings
    assert ROOT_PATH_TO_DATA.exists(), "Data path does not exist."
    assert ROOT_MWA_API_KEY.exists(), "API key is not set."


def load_flare_data(file_path):
    return pd.read_csv(file_path)


def get_downloaded_obs_ids(root_path):
    downloads = os.listdir(root_path)
    return {int(d.split("_")[0]) for d in downloads}


def submit_and_download_jobs_for_new_flare_observations(row, downloaded_ids):
    print(f"Processing flare {row['flare_id']}.")
    start = time.time()

    obs_ids = ast.literal_eval(row['obs_ids'])
    new_obs_ids = [obs_id for obs_id in obs_ids if obs_id not in downloaded_ids]

    if new_obs_ids:
        jobs_to_submit = create_jobs(new_obs_ids)
        if jobs_to_submit:
            process_jobs(jobs_to_submit)

    end = time.time()
    print(f"All tasks completed for flare {row['flare_id']} in {(end - start) / 60:.2f} minutes!")


def create_jobs(obs_ids):
    return [
        (
            'submit_conversion_job_direct',
            {
                'obs_id': obs_id,
                'job_type': 'c',
                'avg_time_res': '4',
                'avg_freq_res': '120',
                'output': 'ms',
            }
        )
        for obs_id in obs_ids
    ]


def process_jobs(jobs):
    params, sslopt, verbose = initialize_settings()
    submit_lock, download_queue, result_queue, status_queue = initialize_queues_and_locks()
    session, jobs_list = login_and_submit_jobs(params, download_queue, status_queue, jobs)
    start_status_thread(status_queue)
    notify = initialize_notifier(params, sslopt, submit_lock, jobs_list, download_queue, result_queue, status_queue, verbose)
    threads = start_download_threads(submit_lock, jobs_list, download_queue, result_queue, status_queue, session)
    results = handle_results(submit_lock, jobs_list, result_queue, download_queue, threads)
    cleanup(notify, threads, result_queue, status_queue, results)
 

def initialize_settings():
    params = (ROOT_SERVER_HTTPS, ROOT_SERVER_URL, ROOT_SERVER_PORT, ROOT_MWA_API_KEY)
    verbose = False
    ssl_verify = os.environ.get("SSL_VERIFY", "0")
    sslopt = {"cert_reqs": ssl.CERT_REQUIRED} if ssl_verify == "1" else {"cert_reqs": ssl.CERT_NONE}
    return params, sslopt, verbose


def initialize_queues_and_locks():
    submit_lock = RLock()
    download_queue = Queue()
    result_queue = Queue()
    status_queue = Queue()
    return submit_lock, download_queue, result_queue, status_queue


def login_and_submit_jobs(params, download_queue, status_queue, jobs_to_submit):
    try:
        session = Session.login(*params)
        status_queue.put("Connected to MWA ASVO.")
    except:
        raise Exception("Could not connect to MWA ASVO.")

    jobs_list = submit_jobs(session, jobs_to_submit, status_queue, download_queue)
    return session, jobs_list


def start_status_thread(status_queue):
    status_thread = Thread(target=status_func, args=(status_queue,))
    status_thread.daemon = True
    status_thread.start()


def initialize_notifier(params, sslopt, submit_lock, jobs_list, download_queue, result_queue, status_queue, verbose):
    try:
        notify = Notify.login(*params, sslopt=sslopt)
        status_queue.put("Connected to MWA ASVO Notifier.")
    except:
        raise Exception("Could not connect to MWA ASVO Notifier.")

    notify_thread = Thread(
        target=notify_func,
        args=(
            notify,
            submit_lock,
            jobs_list,
            download_queue,
            result_queue,
            status_queue,
            verbose,
        ),
    )
    notify_thread.daemon = True
    notify_thread.start()
    return notify


def start_download_threads(submit_lock, jobs_list, download_queue, result_queue, status_queue, session):
    threads = []
    for _ in range(len(jobs_list)):
        t = Thread(
            target=download_func,
            args=(
                submit_lock,
                jobs_list,
                download_queue,
                result_queue,
                status_queue,
                session,
                ROOT_PATH_TO_DATA,
            ),
        )
        threads.append(t)
        t.daemon = True
        t.start()
    return threads


def handle_results(submit_lock, jobs_list, result_queue, download_queue, threads):
    # Handle job results
    results = []
    while True:
        with submit_lock:
            if not jobs_list:
                break

        try:
            r = result_queue.get(timeout=1)
            if not r:
                raise Exception("Error: Control connection lost, exiting")
            results.append(r)
        except Empty:
            continue

    for _ in threads:
        download_queue.put(None)

    for t in threads:
        t.join()

    return results


def cleanup(notify, threads, result_queue, status_queue, results):
    # Cleanup and final steps
    notify.close()
    for t in threads:
        t.join()

    status_queue.put(None)

    while not result_queue.empty():
        r = result_queue.get()
        if r:
            results.append(r)

    if results:
        print("There were errors:")
        for r in results:
            print(f"Error with observation {r.obs_id}; job_id {r.job_id}. Error:")
            print(r.colour_message)


if __name__ == "__main__":
    main()
