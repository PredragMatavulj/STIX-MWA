import pyvo as vo
import pandas as pd
import os


path_to_data = "../data"

tap_service = vo.dal.TAPService("http://vo.mwatelescope.org/mwa_asvo/tap")
query = "SELECT * FROM mwa.observation"
result = tap_service.search(query)
df = result.to_table().to_pandas()
df.to_csv(os.path.join(path_to_data, "mwa_metadata.csv"), index=False)
