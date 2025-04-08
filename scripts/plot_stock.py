import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_trend(file_path, ticker, save_path="reports"):
    df = pd.read_csv(file_path)
    plt.figure(figsize=(10, 4))
    plt.plot(df['Close'], label='Close Price')
    plt.title(f"{ticker} Stock Price Trend")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/{ticker}_trend.png")
    plt.close()