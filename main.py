from scipy.io import loadmat
import pandas as pd
from spsort import getspiketrain, get_psth
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import pickle

if __name__ == "__main__":

	df = pd.read_csv('datasets/10-Sep-2020/info_E1.txt', delim_whitespace=True)

	# how many trials per stimulus
	count = df.groupby('StimName').count()

	d = defaultdict(lambda: defaultdict(list))

	for i, stimname in enumerate(tqdm(df['StimName'].unique())):
		print('Processing {}'.format(stimname))

		stim = df.loc[df['StimName'] == stimname]
		for index, row in stim.iterrows():
			filename = 'Raw_WCWaveform_' + row['date'] + '_E' + str(row['ExpNum']) + '_' + str(row['Trial']) + '.mat'

			data = loadmat(file_name='datasets/10-Sep-2020/' + filename)

			# downsample voltage by 1 kHz
			voltage = data['voltage']

			# call function to get spike train and put in a numpy array
			sptrain = getspiketrain(voltage, 0.01, nclusters=1)
			d[stimname]['sptrain'].append(sptrain)

		# call getPSTH() using the spike trains from all trials
		sps = np.stack(d[stimname]['sptrain'], axis=0)
		psth = get_psth(sps.T.squeeze(), 1000, 10000)
		d[stimname]['meanpsth'].append(psth)


	print("we are done!")

	# output dictionary

	file_name = "../datasets/" + df['date'][0] + "_E" + df['ExpNum'] + "processed.pkl"

	output = open(file_name, 'wb')

	# pickle dictionary using protocol 0
	pickle.dump(d, output)



