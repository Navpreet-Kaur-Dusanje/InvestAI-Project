import os
import pandas as pd
from zipfile import ZipFile

# Path where the Kaggle datasets are downloaded and unzipped
KAGGLE_PATH = "data/raw/kaggle_datasets/"
NEWS_DATA_PATH = "data/raw/stock_news/"
PRICE_VOLUME_PATH = "data/raw/price_volume_data/"

# Function to load the stock news dataset (CSV)
def load_stock_news(news_file="StockNews.csv"):
    news_path = os.path.join(NEWS_DATA_PATH, news_file)
    if os.path.exists(news_path):
        df = pd.read_csv(news_path)
        print(f"Loaded stock news data from {news_path}")
        return df
    else:
        print(f"{news_file} not found!")
        return None

# Function to load and unzip price-volume data
def load_price_volume_data(ticker, zip_file="price-volume-data.zip"):
    zip_path = os.path.join(PRICE_VOLUME_PATH, zip_file)
    extracted_dir = os.path.join(PRICE_VOLUME_PATH, ticker)
    
    if not os.path.exists(extracted_dir):
        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_dir)

    stock_data = pd.read_csv(os.path.join(extracted_dir, f"{ticker}.csv"))
    print(f"Loaded price-volume data for {ticker}")
    return stock_data

# Fetching and combining data for multiple tickers
def get_data(tickers, start_date, end_date):
    all_stock_data = {}
    all_news_data = {}

    for ticker in tickers:
        # Load the news and price data for each ticker
        news_data = load_stock_news(f"{ticker}_news.csv")
        stock_data = load_price_volume_data(ticker)

        # Filter stock data by the desired date range
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        stock_data = stock_data[(stock_data['Date'] >= start_date) & (stock_data['Date'] <= end_date)]

        all_stock_data[ticker] = stock_data
        all_news_data[ticker] = news_data

    return all_stock_data, all_news_data
