import os
import pandas as pd
import nltk
from datetime import timedelta

nltk.download("punkt")


def preprocess_stock_data(stock_data: pd.DataFrame) -> pd.DataFrame:
    stock_data['Close'] = stock_data['Close'].ffill()
    stock_data['Date'] = pd.to_datetime(stock_data['Date'], errors='coerce')
    return stock_data[['Date', 'Close']]


def preprocess_news_articles(news_data: pd.DataFrame) -> pd.DataFrame:
    news_data['content'] = news_data['content'].fillna('')
    news_data['content'] = news_data['content'].apply(nltk.sent_tokenize)
    news_data = news_data.explode('content').reset_index(drop=True)
    news_data['publishedAt'] = pd.to_datetime(news_data['publishedAt'], errors='coerce')
    news_data = news_data.dropna(subset=['publishedAt']).copy()
    news_data['pub_date'] = news_data['publishedAt'].dt.date
    return news_data


def create_finetune_corpus(
    stock_data_dict: dict,
    news_data_dict: dict,
    output_path: str = "data/finetune_corpus.txt"
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for ticker in stock_data_dict:
            stock_df = stock_data_dict[ticker].dropna()
            news_df = news_data_dict[ticker].copy()

            news_df = news_df.dropna(subset=['pub_date'])
            news_df['pub_date'] = pd.to_datetime(news_df['pub_date'], errors='coerce')
            news_df = news_df.dropna(subset=['pub_date'])

            for i in range(len(stock_df)):
                try:
                    stock_date = stock_df.iloc[i, 0]  # Date
                    stock_close = stock_df.iloc[i, 1]  # Close

                    if not pd.api.types.is_datetime64_any_dtype(type(stock_date)):
                        stock_date = pd.to_datetime(stock_date)

                    stock_date = stock_date.date()
                    date_min = pd.to_datetime(stock_date - timedelta(days=1))
                    date_max = pd.to_datetime(stock_date + timedelta(days=1))

                    pub_dates = news_df['pub_date'].dt.normalize().to_numpy()
                    mask = (pub_dates >= date_min) & (pub_dates <= date_max)
                    relevant_news = news_df.loc[mask, 'content']

                    for sentence in relevant_news:
                        if isinstance(sentence, str) and sentence.strip():
                            f.write(f"Stock Trend: {stock_close:.2f}\n")
                            f.write(f"News: {sentence.strip()}\n")
                            f.write("Generate Report:\n\n")

                except Exception as e:
                    print(f"⚠️ Skipped row {i} in {ticker}: {e}")
