import os
from datetime import datetime
from scripts.data_fetcher import get_data
from scripts.data_preprocessor import create_finetune_corpus, preprocess_stock_data, preprocess_news_articles
from scripts.train_model import fine_tune_model
from scripts.generate_report import generate_investment_report

# Set up parameters
TICKERS = ["AAPL", "GOOG", "TSLA"]  # Example tickers, can be modified
START_DATE = "2020-01-01"
END_DATE = "2023-12-31"

# Fetch stock price and news data
print("Fetching data...")
all_stock_data, all_news_data = get_data(TICKERS, START_DATE, END_DATE)

# Preprocess the fetched data
print("Preprocessing data...")
processed_stock_data = {}
processed_news_data = {}
for ticker in TICKERS:
    stock_data = preprocess_stock_data(all_stock_data[ticker])
    news_data = preprocess_news_articles(all_news_data[ticker])
    
    processed_stock_data[ticker] = stock_data
    processed_news_data[ticker] = news_data

# Create the fine-tuning corpus
print("Creating fine-tuning corpus...")
create_finetune_corpus(processed_stock_data, processed_news_data, output_path="data/finetune_corpus.txt")

# Fine-tune the model
print("Fine-tuning the model...")
fine_tune_model(dataset_path="data/finetune_corpus.txt", model_dir="models/investai_gpt2")

# Generate an investment report (example)
print("Generating investment report...")
prompt = "Generate an investment report for Apple stock considering the recent news and price trends."
report = generate_investment_report(prompt, model_dir="models/investai_gpt2")
print("Generated Report:")
print(report)
