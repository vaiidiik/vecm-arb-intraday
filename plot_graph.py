import os
import matplotlib.pyplot as plt
import pandas as pd


def plot_stacked_stocks(csv_path, num_stocks=5):
    # 1. Load the intraday price data
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # 2. Convert Date column to datetime and set as index
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    # 3. Identify the stock columns (excluding index)
    stock_cols = df.columns[:num_stocks]

    # 4. Initialize subplots stacked 1 above another (5 rows, 1 column)
    fig, axes = plt.subplots(
        nrows=num_stocks, ncols=1, figsize=(12, 2 * num_stocks), sharex=True
    )

    # If num_stocks is 1, axes won't be an array, wrap it to handle iteratively
    if num_stocks == 1:
        axes = [axes]

    # 5. Plot each stock in its designated subplot layer
    for i, col in enumerate(stock_cols):
        axes[i].plot(df.index, df[col], label=col, linewidth=1.5)

        # Labels and legends per subplot panel
        axes[i].set_ylabel(f"Price ({col})")
        axes[i].legend(loc="upper left")
        axes[i].grid(True, linestyle="--", alpha=0.5)

    # Set overall formatting
    axes[-1].set_xlabel("Date / Time")
    plt.suptitle("Intraday Stock Price Profiles", fontsize=14, fontweight="bold")

    # Clean layout handling to prevent overlapping text elements
    plt.tight_layout()

    # 6. Display the generated figure
    plt.show()


# Execute the plotting function with your price dataset
if __name__ == "__main__":
    # Ensure the path matches where your script runs relative to the file
    csv_filename = "sample_prices_intraday.csv"
    plot_stacked_stocks(csv_filename, num_stocks=5)