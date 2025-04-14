import numpy as np
import os
import tarfile
import tempfile
import shutil
from helper_functions.casa_compat import import_casatools, import_casatasks
import glob
from casacore.tables import table, taql
import re


tasks = import_casatasks('split','hanningsmooth')
split = tasks.get('split')
hanningsmooth = tasks.get('hanningsmooth')

tools = import_casatools(['tbtool', 'mstool', 'qatool'])

tbtool = tools['tbtool']
mstool = tools['mstool']
qatool = tools['qatool']
tb = tbtool()
ms = mstool()
qa = qatool()

stokestype = [
    'Undefined',
    # undefined value = 0
    'I',
    'Q',
    'U',
    # standard stokes parameters
    'V',
    #
    'RR',
    'RL',
    'LR',
    # circular correlation products
    'LL',
    #
    'XX',
    'XY',
    'YX',
    # linear correlation products
    'YY',
    #
    'RX',
    'RY',
    'LX',
    'LY',
    'XR',
    'XL',
    'YR',
    # mixed correlation products
    'YL',
    #
    'PP',
    'PQ',
    'QP',
    # general quasi-orthogonal correlation products
    'QQ',
    #
    'RCircular',
    'LCircular',
    # single dish polarization types
    'Linear',
    # Polarized intensity ((Q^2+U^2+V^2)^(1/2))
    'Ptotal',
    # Linearly Polarized intensity ((Q^2+U^2)^(1/2))
    'Plinear',
    # Polarization Fraction (Ptotal/I)
    'PFtotal',
    # Linear Polarization Fraction (Plinear/I)
    'PFlinear',
    # Linear Polarization Angle (0.5 arctan(U/Q)) (in radians)
    'Pangle']


stokesenum = {}
for k, v in zip(range(len(stokestype)), stokestype):
    stokesenum[k] = v

def get_dspec(fname=None, specfile=None, bl='', uvrange='', field='', scan='',
                datacolumn='data',
                domedian=False, timeran=None, spw=None, timebin='0s', regridfreq=False,
                hanning=False,
                applyflag=True, fillnan=None, verbose=False,
                usetbtool=True, ds_normalised=False):
    if fname.endswith('/'):
        fname = fname[:-1]
    msfile = fname
    if not spw:
        spw = ''
    if not timeran:
        timeran = ''
    if domedian:
        if not uvrange:
            uvrange = '0.01~5km'
    if not bl:
        bl = ''
    else:
        uvrange = ''
    # Open the ms and plot dynamic spectrum
    if verbose:
        print('Splitting selected data...')

    ms_files, temp_dir = get_ms_files(fname)

    spectrograms, frequencies = [], []
    for i, ms_file in enumerate(ms_files):
        fname = os.path.join(temp_dir, ms_file.name)
        fname = os.path.normpath(fname)

        if usetbtool:
            if datacolumn.lower() == 'data':
                datacol = 'DATA'
                #print('Using DATA column')
            if datacolumn.lower() == 'corrected':
                datacol = 'CORRECTED_DATA'
                #print('Using CORRECTED_DATA column')
            if verbose:
                print('using table tool to extract the data')
            try:
                tb.open(fname + '/POLARIZATION')
                corrtype = tb.getcell('CORR_TYPE', 0)
                pol = [stokesenum[p] for p in corrtype]
                tb.close()
            except:
                pol = []

            if hanning:
                hanningsmooth(vis=fname, datacolumn='data', field=field, outputvis=fname + '.tmpms')
                fname = fname + '.tmpms'
            
            tb.open(fname)
            #tb.open(fname)
            spwtb = tbtool()
            spwtb.open(fname + '/SPECTRAL_WINDOW')
            ptb = tbtool()
            ptb.open(fname + '/POLARIZATION')
            npol = ptb.getcol('NUM_CORR', 0, 1)[0]
            ptb.close()

            ms.open(fname)
            # ms.selectinit(datadescid=0)
            spwlist = []
            mdata = ms.metadata()
            # determine how many unique spws
            # now it ignores the spw parameter and use all of them
            spws_flds = mdata.spwsforfields()
            spws_ = []
            for k in spws_flds.keys():
                spws_.append(spws_flds[k])
            spws_unq = np.unique(spws_)
            nspw = len(spws_unq)
            nbaselines = mdata.nbaselines()
            nantennas = mdata.nantennas()
            scannumbers = mdata.scannumbers()
            spw_nfrq = []  # List of number of frequencies in each spw
            for i in spws_unq:
                spw_nfrq.append(mdata.nchan(i))
            spw_nfrq = np.array(spw_nfrq)
            nf = np.sum(spw_nfrq)
            smry = mdata.summary()

            scanids = sorted(smry['observationID=0']['arrayID=0'].keys())
            ## todo find a way to determine the target fieldID if multi-fields exist
            smryscan0 = smry['observationID=0']['arrayID=0'][scanids[0]]
            fieldid = ''
            for k in smryscan0.keys():
                if k.startswith('fieldID='):
                    fieldid = k
                    break
            nbl = int(smryscan0[fieldid]['0']['nrows'] / nspw)
            if nbl == nbaselines + nantennas:
                hasautocorr = True
            elif nbl == nbaselines:
                hasautocorr = False
            else:
                raise (ValueError('The baseline number is not correct.'))

            antmask = []
            if uvrange != '' or bl != '':
                ms.open(fname)
                ms.selectinit(datadescid=0, reset=True)
                mdata = ms.metadata()
                antlist = mdata.antennaids()
                smry = mdata.summary()
                mdata.done()
                staql = {'uvdist': uvrange, 'baseline': bl, 'spw': spw, 'field': field, 'scan': scan,
                            'timerange': timeran}
                ### todo the selection only works for uvrange and bl. To make the selection of other items works,
                a = ms.msselect(staql)
                mdata = ms.metadata()
                baselines = mdata.baselines()
                if hasautocorr:
                    for lidx, l in enumerate(antlist):
                        antmask.append(baselines[l][antlist[lidx:]])
                else:
                    for lidx, l in enumerate(antlist):
                        antmask.append(baselines[l][antlist[lidx + 1:]])
                antmask = np.hstack(antmask)
                mdata.done()
                ms.close()
                ms.open(fname)

            scan_ntimes = []  # List of number of times in each scan
            nrows = []
            for s, scanid in enumerate(scanids):
                smryscan = smry['observationID=0']['arrayID=0'][scanid]
                nrow = next(v for k, v in smryscan.items() if 'fieldID=' in k)['nrows']
                nrows.append(nrow)
                scan_ntimes.append(nrow / nspw / nbl)
            scan_ntimes = np.array(scan_ntimes)
            scan_ntimes_integer = scan_ntimes.astype(int)
            if len(np.where(scan_ntimes % scan_ntimes_integer != 0)[0]) != 0:
                # if True:
                scan_ntimes = []  # List of number of times in each scan
                for s, scanid in enumerate(scanids):
                    nrows_scan = []  ## get the nrows for each time. They are not always the SAME!
                    smryscan = smry['observationID=0']['arrayID=0'][scanid]
                    # for k, v in smryscan['fieldID=0'].items():
                    #     if isinstance(v,dict):
                    #         nrows_scan.append(v['nrows'])
                    nrows_scan.append(next(v for k, v in smryscan.items() if 'fieldID=' in k)['nrows'])
                    scan_ntimes.append(nrows[s] / max(set(nrows_scan)))
                    # scan_ntimes.append(
                    #     len(smry['observationID=0']['arrayID=0'][scanids[scannumber]]['fieldID=0'].keys()) - 6)
                scan_ntimes = np.array(scan_ntimes).astype(int)
            else:
                scan_ntimes = scan_ntimes_integer

            nt = np.sum(scan_ntimes)
            times = tb.getcol('TIME')
            if times[nbl] - times[0] != 0:
                # This is frequency/scan sort order
                order = 'f'
            elif times[nbl * nspw - 1] - times[0] != 0:
                # This is time sort order
                order = 't'
            else:
                order = 'f'

            freq = np.zeros(nf, float)
            times = np.zeros(nt, float)
            if verbose:
                print("npol, nf, nt, nbl:", npol, nf, nt, nbl)
            if order == 't':
                # specamp = np.zeros((npol, nf, nbl, nt), complex)
                specamp = np.zeros((npol, nf, nbl, nt), dtype=np.complex64)

                flagf = np.zeros((npol, nf, nbl, nt), int)
                for j in range(nt):
                    fptr = 0
                    # Loop over spw
                    for i, sp in enumerate(spws_unq):
                        # Get channel frequencies for this spw (annoyingly comes out as shape (nf, 1)
                        cfrq = spwtb.getcol('CHAN_FREQ', sp, 1)[:, 0]
                        if j == 0:
                            # Only need this the first time through
                            spwlist += [i] * len(cfrq)
                        if i == 0:
                            times[j] = tb.getcol('TIME', nbl * (i + nspw * j), 1)  # Get the time
                        spec_ = tb.getcol(datacol, nbl * (i + nspw * j), nbl)  # Get complex data for this spw
                        flag = tb.getcol('FLAG', nbl * (i + nspw * j), nbl)  # Get flags for this spw
                        nfrq = len(cfrq)
                        # Apply flags
                        if applyflag:
                            if type(fillnan) in [int, float]:
                                spec_[flag] = float(fillnan)
                            else:
                                spec_[flag] = 0.0
                        # Insert data for this spw into larger array
                        specamp[:, fptr:fptr + nfrq, :, j] = spec_
                        flagf[:, fptr:fptr + nfrq, :, j] = flag
                        freq[fptr:fptr + nfrq] = cfrq
                        fptr += nfrq
            else:
                specf = np.zeros((npol, nf, nt, nbl), complex)  # Array indexes are swapped
                # flagf = np.zeros((npol, nf, nt, nbl), int)  # Array indexes are swapped
                # ant1 = np.zeros((nt, nbl), int)  # Array indexes are swapped
                # ant2 = np.zeros((nt, nbl), int)  # Array indexes are swapped
                iptr = 0
                for j, scanid in enumerate(scanids):
                    # Loop over scans
                    s = scan_ntimes[j]
                    s1 = np.sum(scan_ntimes[:j])  # Start time index
                    s2 = np.sum(scan_ntimes[:j + 1])  # End time index
                    if verbose:
                        print('=======Filling up scan #{0:d} frm:{1:d}-{2:d}======='.format(j, s1, s2))
                    for i, sp in enumerate(spws_unq):
                        # Loop over spectral windows
                        f = spw_nfrq[i]
                        f1 = np.sum(spw_nfrq[:i])  # Start freq index
                        f2 = np.sum(spw_nfrq[:i + 1])  # End freq index
                        if verbose:
                            print('Filling up spw #{0:d} chn:{1:d}--{2:d}'.format(i, f1, f2))
                        spec_ = tb.getcol(datacol, iptr, nbl * s)
                        flag = tb.getcol('FLAG', iptr, nbl * s)
                        if j == 0:
                            cfrq = spwtb.getcol('CHAN_FREQ', sp, 1)[:, 0]
                            freq[f1:f2] = cfrq
                            spwlist += [i] * len(cfrq)
                        times[s1:s2] = tb.getcol('TIME', iptr, nbl * s).reshape(s, nbl)[:, 0]  # Get the times
                        # Apply flags
                        if applyflag:
                            if type(fillnan) in [int, float]:
                                spec_[flag] = float(fillnan)
                            else:
                                spec_[flag] = 0.0
                        # Insert data for this spw into larger array
                        specf[:, f1:f2, s1:s2] = spec_.reshape(npol, f, s, nbl)
                        # flagf[:, f1:f2, s1:s2] = flag.reshape(npol, f, s, nbl)
                        # if i==0:
                        #     ant1[s1:s2] = ant1_.reshape(s, nbl)
                        #     ant2[s1:s2] = ant2_.reshape(s, nbl)
                        # Swap the array indexes back to the desired order
                        # except:
                        #     print('error processing spw {}'.format(i))
                        iptr += nbl * s
                specamp = np.swapaxes(specf, 2, 3)
                # flagf = np.swapaxes(flagf, 2, 3)
            # if applyflag:
            #     specamp = np.ma.masked_array(specamp, flagf)
            tb.close()
            spwtb.close()
            ms.close()
            if len(antmask) > 0:
                specamp = specamp[:, :, np.where(antmask)[0], :]
            (npol, nfreq, nbl, ntim) = specamp.shape
            tim = times
            if hanning:
                os.system('rm -rf {}'.format(fname))
        else:
            # Open the ms and plot dynamic spectrum
            if verbose:
                print('Splitting selected data...')
            vis_spl = './tmpms.splitted'
            if os.path.exists(vis_spl):
                os.system('rm -rf ' + vis_spl)

            # split(vis=msfile, outputvis=vis_spl, timerange=timeran, antenna=bl, field=field, scan=scan, spw=spw,
            #       uvrange=uvrange, timebin=timebin, datacolumn=datacolumn)

            try:
                split(vis=msfile, outputvis=vis_spl, datacolumn=datacolumn, timerange=timeran, spw=spw, antenna=bl,
                        field=field, scan=scan, uvrange=uvrange, timebin=timebin)
            except:
                ms.open(msfile, nomodify=True)
                ms.split(outputms=vis_spl, whichcol=datacolumn, time=timeran, spw=spw, baseline=bl, field=field,
                            scan=scan,
                            uvrange=uvrange, timebin=timebin)
                ms.close()

            # if verbose:
            #   print('Regridding into a single spectral window...')
            # print('Reading data spw by spw')

            try:
                tb.open(vis_spl + '/POLARIZATION')
                corrtype = tb.getcell('CORR_TYPE', 0)
                pol = [stokesenum[p] for p in corrtype]
                tb.close()
            except:
                pol = []

            if regridfreq:
                if verbose:
                    print('Regridding into a single spectral window...')
                ms.open(vis_spl, nomodify=False)
                ms.cvel(outframe='LSRK', mode='frequency', interp='nearest')
                ms.selectinit(datadescid=0, reset=True)
                data = ms.getdata(['amplitude', 'time', 'axis_info'], ifraxis=True)
                specamp = data['amplitude']
                freq = data['axis_info']['freq_axis']['chan_freq']
            else:
                if verbose:
                    print('Concatenating visibility data spw by spw')
                ms.open(vis_spl)
                ms.selectinit(datadescid=0, reset=True)
                spwinfo = ms.getspectralwindowinfo()
                specamp = []
                freq = []
                time = []
                if verbose:
                    print('A total of {0:d} spws to fill'.format(len(spwinfo.keys())))
                for n, descid in enumerate(spwinfo.keys()):
                    ms.selectinit(datadescid=0, reset=True)
                    if verbose:
                        print('filling up spw #{0:d}: {1:s}'.format(n, descid))
                    descid = int(descid)
                    ms.selectinit(datadescid=n)  # , reset=True)
                    data = ms.getdata(['amplitude', 'time', 'axis_info'], ifraxis=True)
                    if verbose:
                        print('shape of this spw', data['amplitude'].shape)
                    specamp_ = data['amplitude']
                    freq_ = data['axis_info']['freq_axis']['chan_freq'].squeeze()
                    if len(freq_.shape) > 1:
                        # chan_freq for each datadecid contains the info for all the spws
                        freq = freq_.transpose().flatten()
                    else:
                        freq.append(freq_)
                    time_ = data['time']
                    if fillnan is not None:
                        flag_ = ms.getdata(['flag', 'time', 'axis_info'], ifraxis=True)['flag']
                        if type(fillnan) in [int, float]:
                            specamp_[flag_] = float(fillnan)
                        else:
                            specamp_[flag_] = 0.0
                    specamp.append(specamp_)
                    time.append(time_)
                specamp = np.concatenate(specamp, axis=1)
                try:
                    # if len(freq.shape) > 1:
                    freq = np.concatenate(freq, axis=0)
                except ValueError:
                    pass
                ms.selectinit(datadescid=0, reset=True)
            ms.close()
            if os.path.exists(vis_spl):
                os.system('rm -rf ' + vis_spl)
            (npol, nfreq, nbl, ntim) = specamp.shape
            freq = freq.reshape(nfreq)

            tim = data['time']

        if verbose:
            print('npol, nfreq, nbl, ntime:', (npol, nfreq, nbl, ntim))
        spec = np.swapaxes(specamp, 2, 1)

        if domedian:
            if verbose:
                print('doing median of all the baselines')
            # mask zero values before median
            # spec_masked = np.ma.masked_where(spec < 1e-9, spec)
            # spec_masked2 = np.ma.masked_invalid(spec)
            # spec_masked = np.ma.masked_array(spec, mask=np.logical_or(spec_masked.mask, spec_masked2.mask))
            # spec_med = np.ma.filled(np.ma.median(spec_masked, axis=1), fill_value=0.)
            spec = np.abs(spec)
            if ds_normalised == False:
                # mask zero values before median
                spec_masked = np.ma.masked_where(spec < 1e-9, spec)
                spec_masked2 = np.ma.masked_invalid(spec)
                spec_masked = np.ma.masked_array(spec, mask=np.logical_or(spec_masked.mask, spec_masked2.mask))
                spec_med = np.ma.filled(np.ma.median(spec_masked, axis=1), fill_value=0.)
                # spec_med = np.nanmedian(spec, axis=1)
                nbl = 1
                ospec = spec_med.reshape((npol, nbl, nfreq, ntim))
            else:
                spec_med_time = np.expand_dims(np.nanmedian(spec, axis=3), axis=3)
                spec_normalised = (spec - spec_med_time) / spec_med_time
                spec_med_bl = np.nanmedian(spec_normalised, axis=1)
                nbl = 1
                ospec = spec_med_bl.reshape((npol, nbl, nfreq, ntim))
                ospec = ospec * 1e4
        else:
            ospec = spec

        # take the XX and YY polarisations
        ospec = (ospec[0] + ospec[3]) / 2

        spectrograms.append(np.mean(np.abs(ospec), axis=0))
        frequencies.extend([int(np.round(f / 1e6)) for f in freq]) # convert to MHz

    """
    # Save the dynamic spectral data
    if not specfile:
        specfile = msfile + '.dspec.npz'
    if os.path.exists(specfile):
        os.system('rm -rf ' + specfile)
    np.savez(specfile, spec=ospec, tim=tim, freq=freq,
                timeran=timeran, spw=spw, bl=bl, uvrange=uvrange, pol=pol)
    if verbose:
        print('Median dynamic spectrum saved as: ' + specfile)
    """

    dspec_entity = {
        "spec": np.concatenate(spectrograms, axis=0),
        "freq": frequencies
    }

     # clean up the temporary directory after your operations
    shutil.rmtree(temp_dir)
    delete_log_files()

    return dspec_entity


def delete_log_files():
    casa_log_files = glob.glob("casa-*.log")
    for f in casa_log_files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"could not delete {f}: {e}")


def get_dspec2(fname=None, domedian=True):
    ms_files, temp_dir = get_ms_files(fname)

    spectrograms, frequencies = [], []
    for i, ms_file in enumerate(ms_files):
        fname = os.path.join(temp_dir, ms_file.name)
        fname = os.path.normpath(fname)

         # query the entire DATA column as a 3D array (nfreq × npol)
        result = taql(f"SELECT DATA FROM '{fname}'")
        data = result.getcol("DATA")  # shape: (nrows, nchan, npol)

         # convert complex data to amplitude and extract polarization
        amp = ( np.abs(data[:, :, 0]) + np.abs(data[:, :, 0]) ) / 2
        amp = np.abs(data[:, :, 0])  # shape: (nrows, nchan)
        amp = amp.T  # shape: (nchan, nrows) -> (freq, time)

         # get number of antennas and baselines
        nant, nbl = get_nbl(fname)
        nbl = nbl + nant  # include autocorrelations
        ntime = data.shape[0] // nbl

        freqs = get_frequencies(fname)
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

    dspec_entity = {
        "spec": np.concatenate(spectrograms, axis=0),
        "freq": frequencies
    }

     # clean up the temporary directory after your operations
    shutil.rmtree(temp_dir)
    delete_log_files()

    return dspec_entity


def get_ms_files(fname):
     # extracting the tar file in a temporary directory
    temp_dir = tempfile.mkdtemp()
    with tarfile.open(fname, 'r') as tar:
         # find the MS file
        ms_files = [member for member in tar.getmembers() if member.name.endswith('.ms')]
        if not ms_files:
            raise ValueError("No .ms files found in the tar archive.")
        else:
            tar.extractall(path=temp_dir)
     # sort by the channel number extracted from the filename
    ms_files = sorted(ms_files, key=lambda f: int(re.search(r'ch(\d+)-', f.name).group(1)))

    return ms_files, temp_dir


def get_nbl(ms_path):
    """
    get the number of antennas and baselines in a measurement set.
    """
    ant_table = table(f"{ms_path}/ANTENNA", ack=False)
    nant = len(ant_table)
    ant_table.close()

    # number of unique baselines (cross-correlations only)
    nbl = nant * (nant - 1) // 2
    return nant, nbl


def get_frequencies(ms_path):
    """
    get frequences in a measurement set.
    """
    spw = table(f"{ms_path}/SPECTRAL_WINDOW", ack=False)
    chan_freq = spw.getcol("CHAN_FREQ")[0]  # shape: (nchan,)
    spw.close()

    freqs_mhz = chan_freq / 1e6  # convert Hz to MHz
    return freqs_mhz