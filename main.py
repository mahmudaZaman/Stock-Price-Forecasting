import os
import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from src.features.create_windows import windowed_dataset
from src.models.train_model import run_train_pipeline, ModelTrainer


def preprocess_data(data_df):
    data_df = data_df.set_index('Date')
    # train test split
    train_data = data_df['Close']['2020-03-10':'2021-07-25'].values
    test_data = data_df['Close']['2021-07-25':].values
    S = MinMaxScaler()
    scaled_train = S.fit_transform(train_data.reshape(-1, 1))
    scaled_test = S.transform(test_data.reshape(-1, 1))

    test_set_to_prediction = np.concatenate([scaled_train[-21:], scaled_test], axis=0)
    X_train, y_train = np.array(list(windowed_dataset(scaled_train, 10))[0][0]), np.array(
        list(windowed_dataset(scaled_train, 10))[0][1])

    X_test, y_test = np.array(list(windowed_dataset(test_set_to_prediction, 10))[0][0]), np.array(
        list(windowed_dataset(test_set_to_prediction, 10))[0][1])
    return X_train, y_train,X_test, y_test

def streamlit_run():
    model = tf.keras.models.load_model('/Users/shuchi/Documents/work/personal/Stock-Price-Forecasting/src/models/out/forcasting_model.h5')
    st.title('Stock Price Prediction App')
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        # df = pd.read_csv("/Users/shuchi/Documents/work/personal/Stock-Price-Forecasting/dataset/AAPL.csv")
        df = pd.read_csv(uploaded_file)
        X_train, y_train,X_test, y_test = preprocess_data(df)
        model_trainer = ModelTrainer()
        predictions = model_trainer.initiate_model_trainer(X_train, y_train, X_test, y_test)

        st.subheader("Actual vs Predicted Values:")
        # Plot actual vs predicted values
        q = np.arange(0, len(y_train))
        plt.figure(figsize=(10, 6))
        plt.plot(q, y_train, label='Actual')
        plt.plot(q, predictions, label='Predicted')
        plt.xlabel('Date')
        plt.ylabel('Stock Price (Close)')
        plt.title('Actual vs Predicted Stock Prices (Close)')
        plt.legend()
        st.pyplot(plt)

def model_run():
    run_train_pipeline()


if __name__ == '__main__':
    mode = os.getenv("mode", "streamlit")
    print("mode", mode)
    if mode == "model":
        model_run()
    else:
        streamlit_run()
