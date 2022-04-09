import pandas as pd

import os

def main():
	print('Preprocessing Kaggle dataset...')
	raw_files = [os.path.join(os.getcwd(), f'articles{i}.csv') for i in range(1, 4)]
	
	# Check that Kaggle files are downloaded
	for f in raw_files:
		if not os.path.isfile(f):
			raise FileNotFoundError(f'Missing {f} Did you download the Kaggle dataset?')

	out_df = pd.concat(pd.read_csv(f) for f in raw_files)

	out_df.drop(['Unnamed: 0'], axis=1, inplace=True)
	out_df.to_csv(os.path.join(os.getcwd(), 'dataset.csv'), index=False)
	print('Done.')

if __name__ == '__main__':
	main()