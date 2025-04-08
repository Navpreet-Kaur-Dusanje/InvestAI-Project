import streamlit as st
from scripts.generate_report import generate_investment_report
from scripts.data_fetcher import get_data
from scripts.data_preprocessor import preprocess_stock_data, preprocess_news_articles
from datetime import datetime

# Set up page title
st.title("InvestAI: Personalized Investment Report Generator")

# User input for stock tickers
tickers_input = st.text_input("Enter Stock Tickers (comma separated):", "AAPL, GOOG, TSLA")
tickers = [ticker.strip() for ticker in tickers_input.split(",")]

# User input for date range
start_date = st.date_input("Start Date", datetime(2020, 1, 1))
end_date = st.date_input("End Date", datetime.today())

# Fetch and preprocess data
if st.button("Generate Report"):
    with st.spinner("Fetching and preprocessing data..."):
        # Fetch stock price and news data
        all_stock_data, all_news_data = get_data(tickers, str(start_date), str(end_date))
        
        # Preprocess the fetched data
        processed_stock_data = {}
        processed_news_data = {}
        for ticker in tickers:
            stock_data = preprocess_stock_data(all_stock_data[ticker])
            news_data = preprocess_news_articles(all_news_data[ticker])
            
            processed_stock_data[ticker] = stock_data
            processed_news_data[ticker] = news_data
        
        # Create fine-tuning corpus
        create_finetune_corpus(processed_stock_data, processed_news_data, output_path="data/finetune_corpus.txt")
        
        # Fine-tune the model
        fine_tune_model(dataset_path="data/finetune_corpus.txt", model_dir="models/investai_gpt2")
        
        # Generate report for the first ticker (for simplicity)
        prompt = f"Generate an investment report for {tickers[0]} stock considering the recent news and price trends."
        report = generate_investment_report(prompt, model_dir="models/investai_gpt2")
        
        st.success("Report Generated!")
        st.write(report)
