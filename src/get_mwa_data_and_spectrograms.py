import os
import ssl
import time
from mantaray.scripts.mwa_client import submit_jobs, status_func, download_func, notify_func
from mantaray.api import Session, Notify
from queue import Queue, Empty
from threading import Thread, RLock
from mwa_spectrograms_raw_codes.dspec import get_dspec
import rootutils
from pathlib import Path
from dotenv import dotenv_values

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
    start = time.time()

    obs_ids = ['1355088920']
    jobs_to_submit = [
        (
            'submit_conversion_job_direct',
            {
                'obs_id': obs_ids[0],
                'job_type': 'c',
                'avg_time_res': '4',
                'avg_freq_res': '120',
                'output': 'ms',
            }
        )
    ]

    # Download data
    params, sslopt, verbose = initialize_settings()
    submit_lock, download_queue, result_queue, status_queue = initialize_queues_and_locks()
    session, jobs_list = login_and_submit_jobs(params, download_queue, status_queue, jobs_to_submit)
    start_status_thread(status_queue)
    notify = initialize_notifier(params, sslopt, submit_lock, jobs_list, download_queue, result_queue, status_queue, verbose)
    threads = start_download_threads(submit_lock, jobs_list, download_queue, result_queue, status_queue, session)
    results = handle_results(submit_lock, jobs_list, result_queue, download_queue, threads)
    cleanup(notify, threads, result_queue, status_queue, results)

    # Create spectrograms
    for obs_id, job_id in zip(obs_ids, jobs_list):
        specfile_path = create_spectrograms(f'{obs_id}_{job_id}.ms')

    end = time.time()
    print(f"All tasks completed in {(end - start) / 60} minutes!")


def initialize_settings():
    params = (ROOT_SERVER_HTTPS, ROOT_SERVER_URL, ROOT_SERVER_PORT, ROOT_MWA_API_KEY)
    verbose = False
    ssl_verify = os.environ.get("SSL_VERIFY", "0")
    sslopt = {"cert_reqs": ssl.CERT_REQUIRED} if ssl_verify == "1" else {"cert_reqs": ssl.CERT_NONE}

    return params, sslopt, verbose

def initialize_queues_and_locks():
    # Initialize queues and locks
    submit_lock = RLock()
    download_queue = Queue()
    result_queue = Queue()
    status_queue = Queue()
    return submit_lock, download_queue, result_queue, status_queue

def login_and_submit_jobs(params, download_queue, status_queue, jobs_to_submit):
    # Login and session initialization
    status_queue.put("Connecting to MWA ASVO...")
    session = Session.login(*params)
    status_queue.put("Connected to MWA ASVO")

    if not jobs_to_submit:
        raise Exception("Error: No jobs to submit")

    jobs_list = submit_jobs(session, jobs_to_submit, status_queue, download_queue)
    return session, jobs_list

def start_status_thread(status_queue):
    # Start the status thread
    status_thread = Thread(target=status_func, args=(status_queue,))
    status_thread.daemon = True
    status_thread.start()

def initialize_notifier(params, sslopt, submit_lock, jobs_list, download_queue, result_queue, status_queue, verbose):
    # Start the notifier thread
    status_queue.put("Connecting to MWA ASVO Notifier...")
    notify = Notify.login(*params, sslopt=sslopt)
    status_queue.put("Connected to MWA ASVO Notifier")

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
    # Start download threads
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

def create_spectrograms(measurement):
    file_path = os.path.join(ROOT_PATH_TO_DATA, f'{measurement}.tar')
    save_path = os.path.join(ROOT_PATH_TO_DATA, 'spectrograms')
    ensure_dir(save_path)
    output_specfile = os.path.join(save_path, f'{measurement}.npz')
    specfile_path = get_dspec(fname=file_path, specfile=output_specfile, verbose=True)
    return specfile_path

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == "__main__":
    main()
