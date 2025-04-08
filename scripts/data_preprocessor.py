import os
import pandas as pd
import nltk
from utils.config import *
from scripts.data_fetcher import load_stock_news, load_price_volume_data

nltk.download('punkt')

# Preprocess stock data
def preprocess_stock_data(stock_data):
    stock_data['Close'] = stock_data['Close'].fillna(method='ffill')
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    # We only need the 'Date' and 'Close' columns for fine-tuning
    stock_trends = stock_data[['Date', 'Close']]
    return stock_trends

# Preprocess news articles by breaking them into sentences
def preprocess_news_articles(news_data):
    news_data['content'] = news_data['content'].fillna('')
    news_data['content'] = news_data['content'].apply(lambda x: nltk.sent_tokenize(x))
    # Flatten list of lists into a single list
    news_data = news_data['content'].explode().reset_index(drop=True)
    return news_data

# Combine stock price trends and news for fine-tuning
def create_finetune_corpus(stock_data_dict, news_data_dict, output_path="data/finetune_corpus.txt"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for ticker in stock_data_dict:
            stock_data = stock_data_dict[ticker]
            news_data = news_data_dict[ticker]
            
            # Loop through each date in stock data
            for _, stock_row in stock_data.iterrows():
                stock_date = stock_row['Date']
                stock_close = stock_row['Close']

                # Find the relevant news for that date
                news_on_date = news_data[news_data['publishedAt'].apply(lambda x: pd.to_datetime(x).date() == stock_date.date())]

                # Write stock trend and relevant news to the corpus
                for news in news_on_date:
                    f.write(f"Stock Trend: {stock_close}\n")
                    f.write(f"News: {news}\n")
                    f.write(f"Generate Report:\n\n")
