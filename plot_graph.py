import os
import matplotlib.pyplot as plt
import pandas as pd


def plot_stacked_stocks(csv_path, num_stocks=5):
                                     
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)

                                                         
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

                                                     
    stock_cols = df.columns[:num_stocks]

                                                                       
    fig, axes = plt.subplots(
        nrows=num_stocks, ncols=1, figsize=(12, 2 * num_stocks), sharex=True
    )

                                                                               
    if num_stocks == 1:
        axes = [axes]

                                                        
    for i, col in enumerate(stock_cols):
        axes[i].plot(df.index, df[col], label=col, linewidth=1.5)

                                              
        axes[i].set_ylabel(f"Price ({col})")
        axes[i].legend(loc="upper left")
        axes[i].grid(True, linestyle="--", alpha=0.5)

                            
    axes[-1].set_xlabel("Date / Time")
    plt.suptitle("Intraday Stock Price Profiles", fontsize=14, fontweight="bold")

                                                                
    plt.tight_layout()

                                     
    plt.show()


                                                       
if __name__ == "__main__":
                                                                         
    csv_filename = "sample_prices_intraday.csv"
    plot_stacked_stocks(csv_filename, num_stocks=5)
