from scipy.signal import butter, lfilter, decimate
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def downsample_data(data, sf, target_sf):
	factor = sf/target_sf
	if factor <= 10:
		data_down = decimate(data, factor)
	else:
		factor = 10
		data_down = data
		while factor > 1:
			data_down = decimate(data_down, factor)
			sf = sf/factor
			factor = int(min([10, sf/target_sf]))

	return data_down, sf


def filter_data(data, low, high, sf, order=2):
	# Determine Nyquist frequency
	nyq = sf/2

	# Set bands
	low = low/nyq
	high = high/nyq

	# Calculate coefficients
	b, a = butter(order, [low, high], btype='band')

	# Filter signal
	filtered_data = lfilter(b, a, data)

	return filtered_data


def get_spikes(data, spike_window=80, tf=5, offset=10, max_thresh=350):

	plt.plot(data)
	# Calculate threshold based on data mean
	thresh = np.mean(np.abs(data)) *tf

	# Find positions wherere the threshold is crossed
	pos = np.where(data > thresh)[0]
	pos = pos[pos > spike_window]

	# Extract potential spikes and align them to the maximum
	spike_samp = [pos[0]]
	wave_form = np.empty([1, spike_window*2])
	for i in pos:
		if i < data.shape[0] - (spike_window+1):

			# Data from position where threshold is crossed to end of window
			tmp_waveform = data[i:i+spike_window*2]

			# Check if data in window is below upper threshold (artifact rejection)
			if np.max(tmp_waveform) < max_thresh:
				# Find sample with maximum data point in window
				tmp_samp = np.argmax(tmp_waveform) + i

				# Re-center window on maximum sample and shift it by offset
				tmp_waveform = data[tmp_samp-(spike_window-offset):tmp_samp+(spike_window+offset)]

				# only append data if the difference between the
				# current threshold crossing and the last threshold
				# crossing falls outside the window length
				# to catch all of the spikes, vary the window length to be the
				# minimum ISI
				if tmp_samp - spike_samp[-1] > spike_window:
					spike_samp = np.append(spike_samp, tmp_samp)
					wave_form = np.append(wave_form, tmp_waveform.reshape(-1, spike_window*2), axis=0)

	np.delete(spike_samp, 0)
	wave_form = np.delete(wave_form, 0, axis=0)
	# Remove duplicates
	#ind = np.where(np.diff(spike_samp) > 1)[0]
	#np.delete(ind, np.where(ind <= offset))
	v_peaks = data[spike_samp]

	#spike_samp = spike_samp[ind]
	#wave_form = wave_form[ind]

	plt.plot(data)
	plt.plot(spike_samp, v_peaks, 'ok', markersize=10)

	return spike_samp, wave_form, v_peaks




if __name__ == "__main__":

	data = np.genfromtxt('datasets/testSpikeSortORN.txt', delimiter='\t')
	nt = len(data)

	dt = 0.0001
	# Determine duration of recording in seconds
	dur_sec = int(nt * dt)

	# Create time vector
	time = np.linspace(0, dur_sec, nt)

	# fig, ax = plt.subplots(figsize=(15, 5))
	# ax.plot(time, data, linewidth=0.5)
	# ax.set_title('ORN spiking: sample freq=10KHz')
	# ax.set_xlabel('Time (s)', fontsize=20)
	# plt.show()

	# bandpass filter the data
	spike_data = filter_data(data, low=50, high=4000, sf=1/dt)
	# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 5))
	# ax1.plot(time, data, linewidth=0.5)
	# ax2.plot(time, spike_data, linewidth=0.5)
	#

	spike_samp, wave_form, v_peaks = get_spikes(-spike_data, spike_window=80, offset=40)
	np.random.seed(10)
	fig, ax = plt.subplots(figsize=(15, 5))

	for i in range(100):
		spike = np.random.randint(0, wave_form.shape[0])
		ax.plot(wave_form[spike, :], 'k', alpha=0.1)

	plt.show()


	# reduce number of dimensions with PCA
	scaler = sk.preprocessing.MinMaxScaler()
	dataset_scaled = scaler.fit_transform(wave_form)

	model = PCA(n_components=2)
	#W = model.components_
	pca_result = model.fit_transform(dataset_scaled)
	print("The first two PCs account for {}%".format(np.sum(model.explained_variance_[:2]) * 100))

	# if we want to project new data onto the pca basis, then we just take the scores
	# in each PC and multiply with the waveforms. Now the waveforms are represented in
	# PC space


	#pc_proj = W @ wave_form.T


	num_clus = 6
	kmeans = KMeans(n_clusters=3, random_state=170).fit_predict(pca_result)

	fig, (ax1, ax2) = plt.subplots(1, 2)
	ax1.plot(wave_form[np.where(kmeans == 0)[0], :].T, 'k', alpha=0.1)
	ax1.plot(wave_form[np.where(kmeans == 1)[0], :].T, 'b', alpha=0.1)
	ax1.plot(wave_form[np.where(kmeans == 2)[0], :].T, 'r', alpha=0.1)

	# plot spike trace
	#ax2.plot(-spike_data)

	# overlay circles denoting spikes from different neurons

	# all spikes in unfiltered trace
	raw = data[spike_samp]
	plt.plot(data, linewidth=0.5)
	# all spikes
	#plt.plot(spike_samp, raw, 'o')
	plt.plot(spike_samp[np.where(kmeans == 0)[0]], raw[np.where(kmeans == 0)[0]], 'ok', markersize=2)
	plt.plot(spike_samp[np.where(kmeans == 1)[0]], raw[np.where(kmeans == 1)[0]], 'ob', markersize=2)
	plt.plot(spike_samp[np.where(kmeans == 2)[0]], raw[np.where(kmeans == 2)[0]], 'or', markersize=2)

