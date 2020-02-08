import pandas as pd


class Encoder:
    def __init__(self, data, cat_cols=None):
        if cat_cols is None:
            cat_cols = []
        self.data = data
        self.cat_cols = cat_cols

    def encode(self):
        for col in self.cat_cols:
            self.data.loc[:, col], _ = pd.factorize(self.data[col], sort=True)
        return self.data
