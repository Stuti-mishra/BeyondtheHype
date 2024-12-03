import argparse
import os
import joblib
import pandas as pd
import numpy as np

PREDICTION_WINDOW_MONTHS = [3, 6, 9, 12]  # Constant for this charge-off prediction task.

def aggregate_weekly(df, date_col, group_col, agg_dict):
    """
    Aggregate the data weekly for the given dataframe.
    """
    df['week'] = pd.to_datetime(df[date_col]).dt.to_period('W').apply(lambda r: r.start_time)
    return df.groupby([group_col, 'week']).agg(agg_dict).reset_index()

def process_test_data(account_state_df, payments_df, transactions_df):
    """
    Process test set inputs to prepare features for prediction.
    """
    # Account State Log Features
    account_features = aggregate_weekly(
        account_state_df, 'timestamp', 'agent_id',
        {
            'credit_balance': ['mean', 'min', 'max', 'std'],
            'credit_utilization': ['mean', 'min', 'max', 'std'],
            'interest_rate': ['mean', 'std'],
            'current_missed_payments': ['sum']
        }
    )
    account_features.columns = ['agent_id', 'week'] + [f"account_{col[0]}_{col[1]}" for col in account_features.columns[2:]]

    # Transactions Log Features
    transaction_features = aggregate_weekly(
        transactions_df, 'timestamp', 'agent_id',
        {
            'amount': ['mean', 'sum', 'max'],
            'online': ['sum'],
            'status': lambda x: (x == 'approved').sum() / len(x)  # Approval rate
        }
    )
    transaction_features.columns = ['agent_id', 'week'] + [f"transaction_{col[0]}_{col[1]}" for col in transaction_features.columns[2:]]

    # Payments Log Features
    payment_features = aggregate_weekly(
        payments_df, 'timestamp', 'agent_id',
        {
            'amount': ['mean', 'sum', 'max']
        }
    )
    payment_features.columns = ['agent_id', 'week'] + [f"payment_{col[0]}_{col[1]}" for col in payment_features.columns[2:]]

    # Merge all features
    features = account_features.merge(transaction_features, on=['agent_id', 'week'], how='outer')
    features = features.merge(payment_features, on=['agent_id', 'week'], how='outer')

    # Forward-fill and backward-fill missing weeks for each agent
    features = features.groupby('agent_id').apply(lambda x: x.sort_values('week').ffill().bfill()).reset_index(drop=True)

    # Drop 'week' column as it is no longer needed
    features = features.drop(columns=['week'])
    return features

def main(test_set_dir: str, results_dir: str):
    """
    Main function for loading test set inputs, making predictions, and saving results.
    """
    # Load test set data
    account_state_df = pd.read_csv(os.path.join(test_set_dir, "account_state_log.csv"))
    payments_df = pd.read_csv(os.path.join(test_set_dir, "payments_log.csv"))
    transactions_df = pd.read_csv(os.path.join(test_set_dir, "transactions_log.csv"))

    # Process test set inputs to create features
    test_features = process_test_data(account_state_df, payments_df, transactions_df)
    agent_ids = test_features['agent_id']
    X_test = test_features.drop(columns=['agent_id'])

    # Impute missing values in the test set
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="mean")
    X_test_imputed = imputer.fit_transform(X_test)

    # Initialize output DataFrame
    col_names = {months: f"charge_off_within_{months}_months" for months in PREDICTION_WINDOW_MONTHS}
    output_df = pd.DataFrame(columns=["agent_id"] + list(col_names.values()))
    output_df["agent_id"] = agent_ids

    # Load models and make predictions for each horizon
    for months in PREDICTION_WINDOW_MONTHS:
        model_filename = f"rf_model_{months}.joblib"
        if not os.path.exists(model_filename):
            raise FileNotFoundError(f"Model file {model_filename} not found.")
        
        print(f"Loading model: {model_filename}")
        rf_model = joblib.load(model_filename)

        # Predict probabilities for charge-off
        y_prob = rf_model.predict_proba(X_test_imputed)[:, 1]
        col_name = col_names[months]
        output_df[col_name] = y_prob  # Use probabilities for predictions

    # Save predictions to results.csv
    output_file = os.path.join(results_dir, "results.csv")
    output_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bth_test_set",
        type=str,
        required=True,
        help="Path to the test set directory."
    )
    parser.add_argument(
        "--bth_results",
        type=str,
        required=True,
        help="Path to the results directory."
    )

    args = parser.parse_args()
    main(args.bth_test_set, args.bth_results)