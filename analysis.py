import os
import pandas as pd
import matplotlib.pyplot as plt


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load restaurant sales data from a CSV file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)
    print("Data loaded successfully.")
    print(f"Number of rows: {len(df)}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare the dataset for analysis.
    Expected columns:
    - order_id
    - date
    - item_name
    - category
    - price
    - quantity
    """
    expected_columns = ["order_id", "date", "item_name", "category", "price", "quantity"]
    missing_columns = [col for col in expected_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Remove duplicates
    df = df.drop_duplicates()

    # Drop rows with important missing values
    df = df.dropna(subset=expected_columns)

    # Convert data types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

    # Drop rows that became invalid after conversion
    df = df.dropna(subset=["date", "price", "quantity"])

    # Standardize text columns
    df["item_name"] = df["item_name"].astype(str).str.strip().str.title()
    df["category"] = df["category"].astype(str).str.strip().str.title()

    # Remove impossible values
    df = df[(df["price"] > 0) & (df["quantity"] > 0)]

    # Create revenue column
    df["revenue"] = df["price"] * df["quantity"]

    print("Data cleaned successfully.")
    print(f"Number of rows after cleaning: {len(df)}")
    return df


def save_cleaned_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save cleaned dataset to CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")


def analyze_total_revenue(df: pd.DataFrame) -> float:
    """
    Calculate total revenue.
    """
    total_revenue = df["revenue"].sum()
    print(f"Total Revenue: ${total_revenue:,.2f}")
    return total_revenue


def analyze_top_selling_items(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Find top-selling items by quantity sold.
    """
    top_items = (
        df.groupby("item_name", as_index=False)["quantity"]
        .sum()
        .sort_values(by="quantity", ascending=False)
        .head(top_n)
    )

    print("\nTop Selling Items:")
    print(top_items)
    return top_items


def analyze_daily_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily revenue.
    """
    daily_revenue = (
        df.groupby(df["date"].dt.date, as_index=False)["revenue"]
        .sum()
        .sort_values(by="date")
    )

    print("\nDaily Revenue:")
    print(daily_revenue.head())
    return daily_revenue


def analyze_category_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze total revenue by category.
    """
    category_sales = (
        df.groupby("category", as_index=False)["revenue"]
        .sum()
        .sort_values(by="revenue", ascending=False)
    )

    print("\nRevenue by Category:")
    print(category_sales)
    return category_sales


def plot_top_selling_items(top_items: pd.DataFrame, output_file: str) -> None:
    """
    Create a bar chart of top-selling items.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.bar(top_items["item_name"], top_items["quantity"])
    plt.title("Top Selling Items")
    plt.xlabel("Item Name")
    plt.ylabel("Quantity Sold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    print(f"Saved chart: {output_file}")


def plot_daily_revenue(daily_revenue: pd.DataFrame, output_file: str) -> None:
    """
    Create a line chart for daily revenue.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(daily_revenue["date"], daily_revenue["revenue"], marker="o")
    plt.title("Daily Revenue")
    plt.xlabel("Date")
    plt.ylabel("Revenue ($)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    print(f"Saved chart: {output_file}")


def plot_category_distribution(category_sales: pd.DataFrame, output_file: str) -> None:
    """
    Create a pie chart for category revenue distribution.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    plt.figure(figsize=(8, 8))
    plt.pie(
        category_sales["revenue"],
        labels=category_sales["category"],
        autopct="%1.1f%%",
        startangle=140
    )
    plt.title("Revenue Distribution by Category")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    print(f"Saved chart: {output_file}")


def generate_business_insights(df: pd.DataFrame) -> None:
    """
    Print simple business insights from the data.
    """
    best_item = df.groupby("item_name")["revenue"].sum().idxmax()
    best_item_revenue = df.groupby("item_name")["revenue"].sum().max()

    best_category = df.groupby("category")["revenue"].sum().idxmax()
    best_category_revenue = df.groupby("category")["revenue"].sum().max()

    print("\nBusiness Insights:")
    print(f"- Highest revenue item: {best_item} (${best_item_revenue:,.2f})")
    print(f"- Best performing category: {best_category} (${best_category_revenue:,.2f})")
    print("- Recommendation: Promote top-selling items and focus on high-revenue categories.")
    print("- Recommendation: Use daily revenue trends to plan staffing and inventory.")


def main() -> None:
    input_file = "data/restaurant_sales.csv"
    cleaned_file = "output/cleaned_restaurant_sales.csv"

    top_items_chart = "output/top_selling_items.png"
    daily_revenue_chart = "output/daily_revenue.png"
    category_chart = "output/category_distribution.png"

    try:
        # Load and clean data
        df = load_data(input_file)
        df = clean_data(df)
        save_cleaned_data(df, cleaned_file)

        # Analysis
        analyze_total_revenue(df)
        top_items = analyze_top_selling_items(df)
        daily_revenue = analyze_daily_revenue(df)
        category_sales = analyze_category_sales(df)

        # Charts
        plot_top_selling_items(top_items, top_items_chart)
        plot_daily_revenue(daily_revenue, daily_revenue_chart)
        plot_category_distribution(category_sales, category_chart)

        # Insights
        generate_business_insights(df)

        print("\nProject completed successfully.")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
