import pandas as pd

from sklearn.model_selection import train_test_split

import os

def main():
	print('Preprocessing Kaggle dataset...')
	raw_files = [os.path.join(os.getcwd(), f'articles{i}.csv') for i in range(1, 4)]
	
	# Check that Kaggle files are downloaded
	for f in raw_files:
		if not os.path.isfile(f):
			raise FileNotFoundError('Missing file or did not execute script in `src/data`.')

	# Drop empty title rows
	out_df = pd.concat(pd.read_csv(f) for f in raw_files)
	out_df.drop(['Unnamed: 0'], axis=1, inplace=True)
	nrows_before = out_df.shape[0]
	out_df.dropna(subset=['title'], inplace=True)
	nrows_after = out_df.shape[0]
	print(f'Dropped {nrows_before - nrows_after} null rows.')

	# Create dataset splits 70:15:15
	train_df, val_test_df = train_test_split(out_df, test_size=0.3, random_state=0)
	val_df, test_df = train_test_split(val_test_df, test_size=0.5, random_state=0)

	print(f'Train Data: {train_df.shape[0]} [{train_df.shape[0]/out_df.shape[0] * 100 :.2f}%]')
	print(train_df.head())
	print('-'*40)
	print(f'Validation Data: {val_df.shape[0]} [{val_df.shape[0]/out_df.shape[0] * 100 :.2f}%]')
	print(val_df.head())
	print('-'*40)
	print(f'Test Data: {test_df.shape[0]} [{test_df.shape[0]/out_df.shape[0] * 100 :.2f}%]')	
	print(test_df.head())
	print('-'*40)

	# Write dataset split to csv files
	train_df.to_csv(os.path.join(os.getcwd(), 'train.csv'), index=False)
	val_df.to_csv(os.path.join(os.getcwd(), 'val.csv'), index=False)
	test_df.to_csv(os.path.join(os.getcwd(), 'test.csv'), index=False)
	print('Done.')

if __name__ == '__main__':
	main()