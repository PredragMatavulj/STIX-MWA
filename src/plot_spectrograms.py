import numpy as np
import matplotlib.pyplot as plt


measurements = ['1346990392_823093_ms', '1346990688_823094_ms', '1346990984_823095_ms', '1346991280_823096_ms'] # G0002
measurements = ['1355088920_823088_ms', '1355089040_823089_ms', '1355089160_823090_ms', '1355089280_823091_ms', '1355089400_823092_ms'] # D0043

# Initialize lists to store combined data
combined_spec = []
combined_time = []
freq_axis = None  # Assume frequency axis is consistent across measurements

for measurement in measurements:
    data = np.load(f'/mnt/nas05/data02/predrag/data/mwa_data/spectrograms/{measurement}.npz')

    spec = data['spec']  # Dynamic spectrum array
    freq = data['freq']  # Frequency axis
    time = data['tim']  # Time axis

    # Store the first frequency axis as a reference
    if freq_axis is None:
        freq_axis = freq
    else:
        assert np.array_equal(freq_axis, freq), "Frequency axes do not match across measurements!"

    # Compute the mean magnitude over polarizations and baselines
    mean_spec = np.mean(np.abs(spec), axis=(0, 1))  # Averaging over polarization (axis 0) and baselines (axis 1)

    # Append the data to combined lists
    combined_spec.append(mean_spec)
    combined_time.append(time)



combined_spec = np.hstack(combined_spec)  # Combine spectrograms
combined_time = np.hstack(combined_time)  # Combine time axes

# Plot the combined data
plt.figure(figsize=(12, 6))
plt.imshow(combined_spec, aspect='auto', origin='lower',
           extent=[combined_time.min(), combined_time.max(), freq_axis.min(), freq_axis.max()])
plt.colorbar(label='Intensity')
plt.xlabel('Time [4 s]')
plt.ylabel('Frequency [Hz]')
plt.title('Combined Dynamic Spectrum')
plt.savefig('../files/plots/combined_dynamic_spectrogram_D0043.png')
plt.show()