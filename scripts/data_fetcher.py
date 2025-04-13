import os
import yfinance as yf
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load News API key
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Fetch stock data using yfinance
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data.reset_index(inplace=True)
    return stock_data

# Fetch news using NewsAPI
def fetch_news_data(ticker, from_date, to_date, page_size=50):
    all_articles = []
    url = "https://newsapi.org/v2/everything"
    
    for page in range(1, 3):  # Max 100 results (2 × 50)
        params = {
            "q": ticker,
            "from": from_date,
            "to": to_date,
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": page_size,
            "page": page,
            "apiKey": NEWS_API_KEY
        }

        response = requests.get(url, params=params)
        data = response.json()

        if response.status_code == 200 and data["status"] == "ok":
            all_articles.extend(data["articles"])
        else:
            print(f"Error fetching news for {ticker}: {data}")
            break

    # Convert to DataFrame
    df = pd.DataFrame(all_articles)
    return df

# Wrapper function to fetch both datasets
def get_data(tickers, start_date, end_date):
    all_stock_data = {}
    all_news_data = {}

    for ticker in tickers:
        print(f"Fetching stock data for {ticker}...")
        stock_data = fetch_stock_data(ticker, start_date, end_date)

        print(f"Fetching news articles for {ticker}...")
        news_data = fetch_news_data(ticker, start_date, end_date)

        all_stock_data[ticker] = stock_data
        all_news_data[ticker] = news_data

    return all_stock_data, all_news_data
