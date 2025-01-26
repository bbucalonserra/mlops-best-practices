import pandas as pd
from sklearn import model_selection


def test_expected_columns_function():
    df = pd.read_csv("data/abt.csv", sep=",")
    expected_columns = [
    "dt_ref", "id_customer", 
    "days_first_iteration_life", "days_last_iteration_life", "qty_iterations_life", 
    "current_balance_life", "points_accumulated_life", "negative_points_life", 
    "frequency_life", "points_accumulated_per_day_life", "qty_messages_life", 
    "qty_redemptions_ponies_life", "messages_per_day_life", "pct_transaction_day01_life", 
    "pct_transaction_day02_life", "pct_transaction_day03_life", "pct_transaction_day04_life", 
    "pct_transaction_day05_life", "pct_transaction_day06_life", "pct_transaction_day07_life", 
    "pct_transaction_morning_life", "pct_transaction_afternoon_life", "pct_transaction_night_life", 
    "avg_days_recurrence_life", "median_days_recurrence_life", 
    "days_first_iteration_d7", "days_last_iteration_d7", "qty_iterations_d7", 
    "current_balance_d7", "points_accumulated_d7", "negative_points_d7", 
    "frequency_d7", "points_accumulated_per_day_d7", "qty_messages_d7", 
    "qty_redemptions_ponies_d7", "messages_per_day_d7", "pct_transaction_day01_d7", 
    "pct_transaction_day02_d7", "pct_transaction_day03_d7", "pct_transaction_day04_d7", 
    "pct_transaction_day05_d7", "pct_transaction_day06_d7", "pct_transaction_day07_d7", 
    "pct_transaction_morning_d7", "pct_transaction_afternoon_d7", "pct_transaction_night_d7", 
    "avg_days_recurrence_d7", "median_days_recurrence_d7", 
    "days_first_iteration_d14", "days_last_iteration_d14", "qty_iterations_d14", 
    "current_balance_d14", "points_accumulated_d14", "negative_points_d14", 
    "frequency_d14", "points_accumulated_per_day_d14", "qty_messages_d14", 
    "qty_redemptions_ponies_d14", "messages_per_day_d14", "pct_transaction_day01_d14", 
    "pct_transaction_day02_d14", "pct_transaction_day03_d14", "pct_transaction_day04_d14", 
    "pct_transaction_day05_d14", "pct_transaction_day06_d14", "pct_transaction_day07_d14", 
    "pct_transaction_morning_d14", "pct_transaction_afternoon_d14", "pct_transaction_night_d14", 
    "avg_days_recurrence_d14", "median_days_recurrence_d14", 
    "days_first_iteration_d28", "days_last_iteration_d28", "qty_iterations_d28", 
    "current_balance_d28", "points_accumulated_d28", "negative_points_d28", 
    "frequency_d28", "points_accumulated_per_day_d28", "qty_messages_d28", 
    "qty_redemptions_ponies_d28", "messages_per_day_d28", "pct_transaction_day01_d28", 
    "pct_transaction_day02_d28", "pct_transaction_day03_d28", "pct_transaction_day04_d28", 
    "pct_transaction_day05_d28", "pct_transaction_day06_d28", "pct_transaction_day07_d28", 
    "pct_transaction_morning_d28", "pct_transaction_afternoon_d28", "pct_transaction_night_d28", 
    "avg_days_recurrence_d28", "median_days_recurrence_d28", 
    "churn_flag"
]
    for col in expected_columns:
        assert col in df.columns, f"The expected column {col} isn't found in the dataset."