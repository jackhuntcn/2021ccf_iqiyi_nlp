import json
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

n_splits = 10
seed = 71

train = pd.read_csv('../data/train_scenes.csv', usecols=['id','movie','scene','character','character_name','text','emotions'])
print(train.shape)
print(train.head())

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
folds = {}
count = 0

for train, valid in kfold.split(train['text'], train['emotions']):
    folds['fold_{}'.format(count)] = {}
    folds['fold_{}'.format(count)]['train'] = train.tolist()
    folds['fold_{}'.format(count)]['valid'] = valid.tolist()
    count += 1

print(len(folds) == n_splits)

with open('folds.json', 'w') as fp:
    json.dump(folds, fp)
