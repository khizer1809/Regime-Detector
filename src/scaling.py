

import pandas as pd
from sklearn.preprocessing import StandardScaler


def fit_transform_fold(train: pd.DataFrame, test: pd.DataFrame):

    scaler = StandardScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train),
        index=train.index,
        columns=train.columns,
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test),
        index=test.index,
        columns=test.columns,
    )

    return train_scaled, test_scaled, scaler
