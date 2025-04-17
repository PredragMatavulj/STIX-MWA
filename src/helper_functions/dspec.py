import os
import numpy as np
import re
import tarfile
import tempfile
import shutil
import glob
from casacore.tables import table, taql
from helper_functions.casa_compat import import_casatools
tools = import_casatools(['tbtool', 'mstool'])

tbtool = tools['tbtool']
mstool = tools['mstool']
tb = tbtool()
ms = mstool()

 # clean up CASA log files
casa_log_files = glob.glob("casa-*.log")
for f in casa_log_files:
    try:
        os.remove(f)
    except Exception as e:
        print(f"could not delete {f}: {e}")


def get_dspec(fname, datacolumn='data', domedian=False, uvrange='', fillnan=None, apply_flag = False):
    if fname.endswith('/'):
        fname = fname[:-1]
    if domedian and not uvrange:
        uvrange = '0.01~5km'

    try:
        ms_files, temp_dir = get_ms_files(fname)
    except Exception as e:
        print(f"Error extracting MS files: {e}")

    spectrograms, frequencies = [], []
    try:
        for ms_file in sorted(ms_files, key=lambda f: int(re.search(r'ch(\d+)-', f.name).group(1))):
             # build and normalize file path
            ms_path = os.path.normpath(os.path.join(temp_dir, ms_file.name))

            # choose data column
            datacol = 'DATA' if datacolumn.lower() == 'data' else 'CORRECTED_DATA'

            # read number of polarizations
            tb.open(ms_path + '/POLARIZATION')
            npol = tb.getcol('NUM_CORR')[0]
            tb.close()

            # open spectral window table
            spw_tb = tbtool(); spw_tb.open(ms_path + '/SPECTRAL_WINDOW')

            # get spw IDs
            ms.open(ms_path)
            meta = ms.metadata()
            spws = np.unique([sp for lst in meta.spwsforfields().values() for sp in lst])
            freqs_per_spw = [meta.nchan(sp) for sp in spws]
            nbl = meta.nbaselines()
            nant = meta.nantennas()

            # open main data table
            tb.open(ms_path)
            specamp = np.zeros((npol, sum(freqs_per_spw), nbl, 0), dtype=np.complex64)
            times = []
            fptr = 0  # frequency channel pointer

            # loop over all scans
            scan_ids = sorted(meta.summary()['observationID=0']['arrayID=0'].keys())
            for scan_id in scan_ids:
                scan_sum = meta.summary()['observationID=0']['arrayID=0'][scan_id]
                nrows_scan = next(v for k, v in scan_sum.items() if 'fieldID=' in k)['nrows']
                nbl_full = nbl + nant if domedian else nbl
                nt = nrows_scan // nbl_full  # number of time steps

                # allocate space for one scan block
                spec_block = np.zeros((npol, sum(freqs_per_spw), nbl, nt), dtype=np.complex64)

                # loop over all spws
                for i, sp in enumerate(spws):
                    # get frequencies for this spw
                    cfrq = spw_tb.getcol('CHAN_FREQ', sp, 1)[:, 0]
                    if fptr == 0:
                        frequencies.extend([int(round(f / 1e6)) for f in cfrq])

                    # get data and flags
                    f = freqs_per_spw[i]
                    spec_ = tb.getcol(datacol, fptr, nbl * nt).reshape(npol, f, nt, nbl)
                    flag = tb.getcol('FLAG', fptr, nbl * nt).reshape(npol, f, nt, nbl)

                    # apply flag mask or fill value
                    if apply_flag:
                        if fillnan is not None:
                            spec_[flag] = float(fillnan)
                        else:
                            spec_[flag] = 0.0

                    # insert into full array
                    spec_block[:, sum(freqs_per_spw[:i]):sum(freqs_per_spw[:i + 1]), :, :] = np.swapaxes(spec_, 2, 3)
                    fptr += nbl * nt

                # concatenate scan block
                specamp = np.concatenate((specamp, spec_block), axis=3)
                times.extend(tb.getcol('TIME', 0, nbl * nt)[::nbl])

            tb.close(); spw_tb.close(); ms.close()

            # swap baseline and frequency axes
            spec = np.swapaxes(specamp, 2, 1)  # (npol, nbl, nfreq, nt)

            # compute median across baselines
            if domedian:
                spec = np.abs(spec)
                spec_masked = np.ma.masked_where(spec < 1e-9, spec)
                spec_masked = np.ma.masked_invalid(spec_masked)
                spec_med = np.ma.median(spec_masked, axis=1)  # masked median
                ospec = np.ma.filled(spec_med, fill_value=np.nan).reshape(npol, 1, -1, spec.shape[3])
            else:
                ospec = spec

            # average XX and YY polarizations
            ospec = (ospec[0] + ospec[3]) / 2

            # log scale specrogram
            ospec = np.ma.masked_invalid(ospec)
            ospec = np.ma.log2(np.ma.clip(ospec, 1, None))

            # append spectrogram slice
            spectrograms.append(np.mean(np.abs(ospec), axis=0))  # shape: (nfreq, nt)
    finally:
         # cleanup
        shutil.rmtree(temp_dir)

    dspec_entity = {
        "spec": np.concatenate(spectrograms, axis=0), # concatenate all spectrograms along frequency
        "freq": frequencies
    }
    return dspec_entity


def get_dspec2(fname=None, domedian=True):
    try:
        ms_files, temp_dir = get_ms_files(fname)
    except Exception as e:
        print(f"Error extracting MS files: {e}")
    spectrograms, frequencies = [], []

    try:
        for ms_file in ms_files:
            ms_path = os.path.join(temp_dir, ms_file.name)
            ms_path = os.path.normpath(ms_path)

             # query the entire DATA column as a 3D array (nfreq × npol)
            result = taql(f"SELECT DATA FROM '{ms_path}'")
            data = result.getcol("DATA")  # shape: (nrows, nchan, npol)

             # convert complex data to amplitude and extract polarization
            amp = ( np.abs(data[:, :, 0]) + np.abs(data[:, :, 0]) ) / 2
            amp = np.abs(data[:, :, 0])  # shape: (nrows, nchan)
            amp = amp.T  # shape: (nchan, nrows) -> (freq, time)

             # get number of baselines
            nbl = get_nbl(ms_path)
            ntime = data.shape[0] // nbl

            freqs = get_frequencies(ms_path)
            frequencies.extend([int(np.round(f)) for f in freqs])

             # reshape and process
            amp = ( np.abs(data[:, :, 0]) + np.abs(data[:, :, 3])) / 2  # average polarizations 0 and 3
            amp = amp.reshape((ntime, nbl, -1))  # (time, baseline, freq)

             # mask very small and invalid values
            amp = np.ma.masked_where(amp < 1e-9, amp)
            amp = np.ma.masked_invalid(amp)

             # average over baselines
            if domedian:
                ospec = np.ma.median(amp, axis=1).T  # shape: (freq, time)
            else:
                ospec = np.ma.mean(amp, axis=1).T

             # log scale specrogram
            ospec = np.ma.masked_invalid(ospec)
            ospec = np.ma.log2(np.ma.clip(ospec, 1, None))

            spectrograms.append(ospec)

    finally:
         # clean up the temporary directory after your operations
        shutil.rmtree(temp_dir)

    dspec_entity = {
        "spec": np.concatenate(spectrograms, axis=0),
        "freq": frequencies
    }

    return dspec_entity


def get_ms_files(fname):
    """
    extracts ms files from a tar archive and returns them sorted by channel number if available.
    """
    temp_dir = tempfile.mkdtemp()
    with tarfile.open(fname, 'r') as tar:
        ms_files = [member for member in tar.getmembers() if member.name.endswith('.ms')]
        if not ms_files:
            raise ValueError("No .ms files found in the tar archive.")
        else:
            tar.extractall(path=temp_dir)
     # sort by the channel number extracted from the filename
    ms_files = sorted(ms_files, key=lambda f: int(re.search(r'ch(\d+)(?:-|\.ms)', f.name).group(1)))

    return ms_files, temp_dir


def get_nbl(ms_path):
    """
    get the number of antennas and baselines in a measurement set.
    """
    ant_table = table(f"{ms_path}/ANTENNA", ack=False)
    nant = len(ant_table)
    ant_table.close()

    nbl = nant * (nant - 1) // 2 # number of unique baselines (cross-correlations only)
    nbl += nant # include autocorrelations
    return nbl


def get_frequencies(ms_path):
    """
    get frequences in a measurement set.
    """
    spw = table(f"{ms_path}/SPECTRAL_WINDOW", ack=False)
    chan_freq = spw.getcol("CHAN_FREQ")[0]  # shape: (nchan,)
    spw.close()

    freqs_mhz = chan_freq / 1e6  # convert Hz to MHz
    return freqs_mhz