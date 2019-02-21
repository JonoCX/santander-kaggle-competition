import pandas as pd 

data = pd.read_csv('../data/train.csv')

# what's the distribution of the target
print(data['target'].value_counts())
print(data['target'].value_counts(normalize=True))

