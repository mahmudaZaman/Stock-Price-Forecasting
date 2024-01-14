from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:
    data_uri: str = "/Users/shuchi/Documents/work/personal/Stock-Price-Forecasting/dataset/AAPL.csv"
    print("train_data_uri", data_uri)


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        data_df = pd.read_csv(self.ingestion_config.data_uri)
        print(data_df.head())
        data_df = data_df.set_index('Date')
        # train test split
        train_data = data_df['Close']['2020-03-10':'2021-07-25'].values
        test_data = data_df['Close']['2021-07-25':].values
        return (
            train_data, test_data
        )
