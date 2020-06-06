import os
import argparse
import collections

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, GridSearchCV
from fastFM import als
from implicit.als import AlternatingLeastSquares

from recommender_system_tutorial.types import ImplicitFeedback


def iter_feedbacks(filename):
    with open(filename) as f:
        for line in f:
            fields = line.strip().split('\t')
            yield ExplicitFeedback(*fields)


def iter_implicit_feedbacks(filename):
    with open(filename) as f:
        for line in f:
            fields = line.strip().split('\t')
            user_id = int(fields[0])
            item_id = int(fields[1])
            yield ImplicitFeedback(item_id, user_id)


class ExplicitFeedback(object):

    def __init__(self, user_id, item_id, rating, timestamp):
        self.user_id = user_id
        self.item_id = item_id
        self.rating = int(rating)
        self.timestamp = timestamp


class Encoder(object):

    def __init__(self):
        self.id2index = {}
        self.item_attributes = {}
        self.user_attributes = {}
    
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
            if feedback.item_id in self.item_attributes:
                for name, value in self.item_attributes[feedback.item_id].items():
                    j = self._get_index('item-attribute-' + name, test)
                    if j is not None:
                        rows.append(i)
                        cols.append(j)
                        data.append(value)
            if feedback.user_id in self.user_attributes:
                for name, value in self.user_attributes[feedback.user_id].items():
                    j = self._get_index('user-attribute-' + name, test)
                    if j is not None:
                        rows.append(i)
                        cols.append(j)
                        data.append(value)
            y.append(feedback.rating)
        X = csr_matrix((data, (rows, cols)), shape=(i + 1, len(self.id2index)))
        y = np.array(y)
        return X, y

    def load_item_attributes(self, filename):
        with open(filename, 'rb') as f:
            for line in f:
                fields = line.strip().split(b'|')
                item_id = fields[0].decode('utf-8')
                self.item_attributes[item_id] = {f'category-{category_number}': 1
                                                 for category_number, category in enumerate(fields[5:], start=1) if category == b'1'}

    def load_user_attributes(self, filename):
        with open(filename, 'rb') as f:
            for line in f:
                fields = line.strip().split(b'|')
                user_id = fields[0].decode('utf-8')
                occupation = fields[3].decode('utf-8')
                self.user_attributes[user_id] = {
                    'age': int(fields[1]),
                    'gender': 1 if fields[2] == b'M' else 0,
                    f'occupation-{occupation}': 1
                }


    def _get_index(self, key, test=False):
        if test:
            if key in self.id2index:
                return self.id2index[key]
            else:
                return None
        else:
            return self.id2index.setdefault(key, len(self.id2index))


def explicit(args):
    encoder = Encoder()
    encoder.load_item_attributes(os.path.join(args.in_dir, 'u.item'))
    encoder.load_user_attributes(os.path.join(args.in_dir, 'u.user'))
    X, y = encoder.get_Xy(os.path.join(args.in_dir, 'ua.base'))
    print(X.shape)
    print(len(y))
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


def implicit(args):
    row_dict, col_dict = {}, {}
    rows, cols, data = [], [], []
    for feedback in iter_implicit_feedbacks(os.path.join(args.in_dir, 'ua.base')):
        i = row_dict.setdefault(feedback.item_id, len(row_dict))
        j = col_dict.setdefault(feedback.user_id, len(col_dict))
        rows.append(i)
        cols.append(j)
        data.append(1)
    item_user_data = csr_matrix((data, (rows, cols)), shape=(len(row_dict), len(col_dict)))

    model = AlternatingLeastSquares(factors=8)
    model.fit(item_user_data)

    # Evaluation
    user_items = item_user_data.T.tocsr()
    user_items_test = collections.defaultdict(set)
    for feedback in iter_implicit_feedbacks(os.path.join(args.in_dir, 'ua.test')):
        try:
            i = row_dict[feedback.item_id]
            j = col_dict[feedback.user_id]
        except KeyError as e:
            continue
        user_items_test[j].add(i)
    
    topk = 10
    precision = 0
    for user_index, item_indices in user_items_test.items():
        recommendations = model.recommend(user_index, user_items, topk, True)
        precision += sum(1 if item_index in item_indices else 0 for item_index, _ in recommendations) / topk
    precision = precision / len(user_items_test)
    print('precision:', precision)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dir', required=True)
    parser.add_argument('--random-state', type=int, default=1)
    subparsers = parser.add_subparsers()
    p = subparsers.add_parser('explicit')
    p.set_defaults(main=explicit)
    p = subparsers.add_parser('implicit')
    p.set_defaults(main=implicit)
    return parser


if __name__ == "__main__":
    p = get_parser()
    args = p.parse_args()
    args.main(args)
