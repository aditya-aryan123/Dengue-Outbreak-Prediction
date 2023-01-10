import pandas as pd


df = pd.read_csv('../input/train_folds_updated.csv')

feat = pd.read_csv('features.txt', sep=" ", header=None)
feat.columns = ['ID', 'Column Name']
feat_to_c = feat['Column Name']
feat_list = feat_to_c.tolist()

print(df.shape)
updated_frame = df.drop(feat_list, axis=1)
print(updated_frame.shape)
updated_frame.to_csv('updated_frame.csv', index=False)

