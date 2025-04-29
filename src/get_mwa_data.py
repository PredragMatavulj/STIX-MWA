import os
import ast
import pandas as pd
import find_flares_in_mwa
from helper_functions.utils import get_root_path_to_data
from helper_functions.mwa_asvo import create_jobs, process_mwa_asvo_jobs
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    """
    download mwa data based on provided observation ids or flares matched with mwa metadata.
    """
    info_path = '../info'
    observation_ids = ['1126847624']  # set to [] to use flare list

    if observation_ids:
        download_by_obs_ids(observation_ids)
    else:
        download_by_flare_overlap(info_path, flare_range=(2500, 3000))


def download_by_obs_ids(observations):
    """
    downloads mwa data using a manual list of observation ids
    """
    download_mwa_data(observations, path_to_data=get_root_path_to_data())


def download_by_flare_overlap(info_path, flare_range=None):
    """
    downloads mwa data using flares overlapping with mwa observation times
    """
    for use_time_corrected in [True, False]:
        filename = (
            "flares_recorded_by_mwa_with_time_correction.csv"
            if use_time_corrected else
            "flares_recorded_by_mwa_no_time_correction.csv"
        )
        flarelist_path = os.path.join(info_path, filename)

        # auto-generate flare file if missing
        if not os.path.exists(flarelist_path):
            find_flares_in_mwa.main()

        flare_data = pd.read_csv(flarelist_path)

        for i, row in flare_data.iterrows():
            if flare_range is None or (flare_range[0] <= i < flare_range[1]):
                download_mwa_data(row, path_to_data=get_root_path_to_data(), is_flare_row=True)


def download_mwa_data(obs_source, path_to_data, avg_time_res=4, avg_freq_res=160, is_flare_row=False):
    """
    downloads mwa data based on observation ids or flare row
    - obs_source: list of obs_ids or a flare row with 'obs_ids' field
    - path_to_data: path where downloaded data are stored
    - avg_time_res: time averaging resolution
    - avg_freq_res: frequency averaging resolution
    - is_flare_row: set to True if passing a flare row
    """
     # a workaround to avoid an error
    change_websocket_abnf()
    
    if is_flare_row:
        obs_ids = ast.literal_eval(obs_source['obs_ids'])
        flare_id = obs_source.get('flare_id', 'unknown')
    else:
        obs_ids = obs_source
        flare_id = None

    downloaded_ids = get_downloaded_obs_ids(path_to_data)
    new_obs_ids = [obs_id for obs_id in obs_ids if obs_id not in downloaded_ids]

    if not new_obs_ids:
        if flare_id:
            logging.info(f"All observations for flare {flare_id} have already been downloaded.")
        else:
            logging.info(f"All observations {new_obs_ids} have already been downloaded.")
        return

    jobs_to_submit = create_jobs(new_obs_ids, avg_time_res, avg_freq_res)

    if jobs_to_submit:
        if flare_id:
            logging.info(f"Submitting {len(jobs_to_submit)} jobs for flare {flare_id}.")
        else:
            logging.info(f"Submitting {len(jobs_to_submit)} jobs for observations {new_obs_ids}.")
        process_mwa_asvo_jobs(jobs_to_submit)


def change_websocket_abnf():
    """
    This is a workaround to avoid the error "websocket._abnf.ABNF.validate() got an unexpected keyword argument 'skip_utf8_validation'"
    when using the websocket library.
    """
    import websocket._abnf
    original_validate = websocket._abnf.ABNF.validate
    def patched_validate(self, skip_utf8_validation):
        self.rsv1 = 0
        self.rsv2 = 0
        self.rsv3 = 0
        return  # skip everything else
    websocket._abnf.ABNF.validate = patched_validate


def get_downloaded_obs_ids(root_path):
    downloads = os.listdir(root_path)
    return {int(d.split("_")[0]) for d in downloads}


if __name__ == "__main__":
    main()
