import tensorflow as tf
from src.features.data_ingestion import DataIngestion
from src.features.data_transformation import DataTransformation
from keras.models import Sequential
from keras.layers import LSTM,Dense

class ModelTrainer:
    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        model = Sequential(
            [
                LSTM(units=60, input_shape=(X_train.shape[1], 1), return_sequences=True),
                LSTM(units=50, activation="relu", return_sequences=True),
                LSTM(units=30, activation="relu", return_sequences=True),
                LSTM(units=20, activation="relu", return_sequences=True),
                LSTM(10),
                Dense(units=1),
            ])

        model.compile(loss="mae",
                       optimizer=tf.keras.optimizers.Adam(),
                       metrics=["mae"])

        callbacks = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            mode='max',
            patience=10,
            verbose=1
        )
        model.fit(X_train, y_train, epochs=300, callbacks=[callbacks])
        output = model.predict(X_train)
        model.evaluate(y_train, output)
        model.save("out/forcasting_model.h5")
        return output


def run_train_pipeline():
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    X_train, y_train, X_test, y_test = data_transformation.initiate_data_transformation(train_data, test_data)
    model_trainer = ModelTrainer()
    predicted_values = model_trainer.initiate_model_trainer(X_train, y_train, X_test, y_test)
    print("predicted values", predicted_values[:5])
    print("model training completed")


# if __name__ == '__main__':
#     run_train_pipeline()