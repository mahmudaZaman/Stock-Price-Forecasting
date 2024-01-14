import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src.features.create_windows import windowed_dataset


class DataTransformation:
    @staticmethod
    def initiate_data_transformation(train_data, test_data):
        S = MinMaxScaler()
        scaled_train = S.fit_transform(train_data.reshape(-1, 1))
        scaled_test = S.transform(test_data.reshape(-1, 1))

        test_set_to_prediction = np.concatenate([scaled_train[-21:], scaled_test], axis=0)
        X_train, y_train = np.array(list(windowed_dataset(scaled_train, 10))[0][0]), np.array(
            list(windowed_dataset(scaled_train, 10))[0][1])

        X_test, y_test = np.array(list(windowed_dataset(test_set_to_prediction, 10))[0][0]), np.array(
            list(windowed_dataset(test_set_to_prediction, 10))[0][1])
        return (
            X_train, y_train, X_test, y_test
        )
