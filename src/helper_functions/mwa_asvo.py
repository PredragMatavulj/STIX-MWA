import os
import ssl
import rootutils
from pathlib import Path
from dotenv import load_dotenv
from queue import Queue, Empty
from threading import Thread, RLock
from helper_functions.utils import get_root_path_to_data
from mantaray.api import Session, Notify
from mantaray.scripts.mwa_client import submit_jobs, status_func, download_func, notify_func


def create_jobs(observations, time_resolution=4, freq_resolution=160):
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
    data_path = get_root_path_to_data()
    params, sslopt, verbose = initialize_settings()
    submit_lock, download_queue, result_queue, status_queue = initialize_queues_and_locks()
    session, jobs_list = login_and_submit_jobs(params, download_queue, status_queue, jobs)
    start_status_thread(status_queue)
    notify = initialize_notifier(params, sslopt, submit_lock, jobs_list, download_queue, result_queue, status_queue, verbose)
    threads = start_download_threads(submit_lock, jobs_list, download_queue, result_queue, status_queue, session, data_path)
    results = handle_results(submit_lock, jobs_list, result_queue, download_queue, threads)
    cleanup(notify, threads, result_queue, status_queue, results)


def initialize_settings():
    """Sets up the necessary environment for the project."""
     # setup project root and load environment variables
    rootutils.setup_root(Path(__file__).resolve(), indicator=".project-root", pythonpath=True)
    load_dotenv()

    SERVER_URL = os.getenv("SERVER_URL", "asvo.mwatelescope.org")
    SERVER_PORT = os.getenv("SERVER_PORT", "443")
    SERVER_HTTPS = os.getenv("SERVER_HTTPS", "1")
    MWA_API_KEY = os.getenv("MWA_API_KEY", "default_api_key")

    params = (SERVER_HTTPS, SERVER_URL, SERVER_PORT, MWA_API_KEY)
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


def start_download_threads(submit_lock, jobs_list, download_queue, result_queue, status_queue, session, data_path):
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
                data_path,
            ),
        )
        threads.append(t)
        t.daemon = True
        t.start()
    return threads


def handle_results(submit_lock, jobs_list, result_queue, download_queue, threads):
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
    notify.close()
    for t in threads:
        t.join()

    status_queue.put(None)

    while not result_queue.empty():
        r = result_queue.get()
        if r:
            results.append(r)

    if results:
        for r in results:
            print(f"Error with observation {r.obs_id}; job_id {r.job_id}. Error: {r.colour_message}")
