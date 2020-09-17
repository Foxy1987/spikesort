from scipy.signal import butter, lfilter, decimate
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering

def get_psth(sptrain, winsize, samprate):
	smooth_win = np.hanning(winsize) / np.hanning(winsize).sum()
	# for multiple trials, average spikes across time bins and then convolve
	psth = np.convolve(np.mean(sptrain, axis=1), smooth_win)
	sigrange = np.arange(winsize // 2, winsize // 2 + len(sptrain))
	psth_ = psth[sigrange]*samprate
	return psth_


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

	# Set freq
	high = high/nyq

	# Calculate coefficients
	b, a = butter(order, high, btype='high')

	# Filter signal
	filtered_data = lfilter(b, a, data)

	return filtered_data


def get_spikes(data, spike_window=80, tf=5, offset=10, max_thresh=350):

	#plt.plot(data)
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

	#plt.plot(data)
	#plt.plot(spike_samp, v_peaks, 'ok', markersize=10)

	return spike_samp, wave_form, v_peaks




def getspiketrain(data, dt, nclusters):
	plot = 0

	nt = len(data)

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
	spike_data = filter_data(data, low=50, high=20, sf=1/dt)
	# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 5))
	# ax1.plot(time, data, linewidth=0.5)
	# ax2.plot(time, spike_data, linewidth=0.5)
	#

	spike_samp, wave_form, v_peaks = get_spikes(-spike_data, spike_window=40, offset=10)
	#np.random.seed(10)
	#fig, ax = plt.subplots(figsize=(5, 8))

	#for i in range(100):
		#spike = np.random.randint(0, wave_form.shape[0])
		#ax.plot(wave_form[spike, :], 'k', alpha=0.1)
	#plt.show()


	# reduce number of dimensions with PCA
	scaler = sk.preprocessing.StandardScaler()
	dataset_scaled = scaler.fit_transform(wave_form)

	model = PCA()
	pca_result = model.fit_transform(dataset_scaled)

	# if we want to project new data onto the pca basis, then we just take the scores
	# in each PC and multiply with the waveforms. Now the waveforms are represented in
	# PC space

	agg = AgglomerativeClustering(n_clusters=nclusters)
	assignment = agg.fit_predict(pca_result)

	if plot:
		fig = plt.figure(constrained_layout=True)
		gs = fig.add_gridspec(2, 2)
		f_ax1 = fig.add_subplot(gs[0, 0])

		f_ax2 = fig.add_subplot(gs[0, 1])

		f_ax3 = fig.add_subplot(gs[1, :])
		plt.tight_layout()


		f_ax1.scatter(pca_result[:, 0], pca_result[:, 1], c=assignment)


		colors = ['#441975', 'y', '#03ADD5']
		for i in range(nclusters):
			f_ax2.plot(wave_form[np.where(assignment == i)[0], :].T, color=colors[i], alpha=0.3)

		# overlay circles denoting spikes from different neurons

		# all spikes in unfiltered trace
		raw = data[spike_samp]
		f_ax3.plot(time, data, '-k', linewidth=0.5)
		# all spikes
		#plt.plot(spike_samp, raw, 'o')
		for i in range(nclusters):
			f_ax3.plot(time[spike_samp[np.where(assignment == i)[0]]], raw[np.where(assignment == 0)[0]], 'o', color=colors[i], markersize=5)

	# get spike times of each cluster
	sptimes1 = time[spike_samp[np.where(assignment == 0)[0]]]

	# bin the spike times
	sp_count_fun = lambda x: np.histogram(x, np.arange(0.5, len(data) + 1) * dt - dt)[0]
	sps = sp_count_fun(sptimes1)
	sps = sps.reshape((-1, 1))

	return sps
