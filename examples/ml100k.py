import os
import argparse

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, GridSearchCV
from fastFM import als


def iter_feedbacks(filename):
    with open(filename) as f:
        for line in f:
            fields = line.strip().split('\t')
            yield ExplicitFeedback(*fields)


class ExplicitFeedback(object):

    def __init__(self, user_id, item_id, rating, timestamp):
        self.user_id = user_id
        self.item_id = item_id
        self.rating = int(rating)
        self.timestamp = timestamp


class Encoder(object):

    def __init__(self):
        self.id2index = {}
    
    def get_Xy(self, filename, test=False):
        rows, cols, data = [], [], []
        y = []
        for i, feedback in enumerate(iter_feedbacks(filename)):
            j = self._get_index('user-' + feedback.user_id, test)
            if j is not None:
                rows.append(i)
                cols.append(j)
                data.append(1)
            j = self._get_index('item-' + feedback.item_id, test)
            if j is not None:
                rows.append(i)
                cols.append(j)
                data.append(1)
            y.append(feedback.rating)
        X = csr_matrix((data, (rows, cols)), shape=(i + 1, len(self.id2index)))
        y = np.array(y)
        return X, y
    
    def _get_index(self, key, test=False):
        if test:
            if key in self.id2index:
                return self.id2index[key]
            else:
                return None
        else:
            return self.id2index.setdefault(key, len(self.id2index))


def main(args):
    encoder = Encoder()
    X, y = encoder.get_Xy(os.path.join(args.in_dir, 'ua.base'))
    fm = als.FMRegression(random_state=args.random_state)
    
    # cross-validation
    param_grid = {
        'rank': [2, 4, 8, 16]
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=args.random_state)
    gs = GridSearchCV(fm, param_grid, scoring='neg_mean_squared_error', cv=cv)
    gs.fit(X, y)
    fm = gs.best_estimator_
    
    X_test, y_test = encoder.get_Xy(os.path.join(args.in_dir, 'ua.test'), test=True)
    y_pred = fm.predict(X_test)
    print(np.c_[y_test, y_pred][:10])

    mse = mean_squared_error(y_test, y_pred)
    print(f'RMSE: {np.sqrt(mse)}')


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--in-dir', required=True)
    p.add_argument('--random-state', type=int, default=1)
    return p


if __name__ == "__main__":
    p = get_parser()
    args = p.parse_args()
    main(args)
