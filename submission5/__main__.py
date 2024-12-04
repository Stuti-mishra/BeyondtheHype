import argparse
import os
import pandas as pd
import joblib

PREDICTION_WINDOW_MONTHS = [3, 6, 9, 12]  # Constant for this charge-off prediction task

# Enforce cascading logic across weeks for predictions
def enforce_cascading_weekly_logic(weekly_data):
    """
    Enforce cascading logic for weekly predictions across all weeks for the same agent.
    """
    weekly_data = weekly_data.sort_values(['agent_id', 'week'])
    cascading_results = []
    for agent_id, group in weekly_data.groupby('agent_id'):
        group = group.copy()
        max_class = 0
        for index, row in group.iterrows():
            max_class = max(max_class, row['predicted_class'])
            group.loc[index, 'charge_off_within_3_months'] = int(max_class >= 1)
            group.loc[index, 'charge_off_within_6_months'] = int(max_class >= 2)
            group.loc[index, 'charge_off_within_9_months'] = int(max_class >= 3)
            group.loc[index, 'charge_off_within_12_months'] = int(max_class >= 4)
        cascading_results.append(group)
    return pd.concat(cascading_results)

# Aggregate weekly predictions into final agent-level predictions
def aggregate_final_predictions(cascading_data):
    """
    Aggregate weekly predictions into final predictions per agent.
    """
    final_results = cascading_data.groupby('agent_id').agg({
        'charge_off_within_3_months': 'max',
        'charge_off_within_6_months': 'max',
        'charge_off_within_9_months': 'max',
        'charge_off_within_12_months': 'max'
    }).reset_index()
    return final_results

def main(test_set_dir: str, results_dir: str):
    # Load test set data
    account_state_df = pd.read_csv(os.path.join(test_set_dir, "account_state_log.csv"))
    payments_df = pd.read_csv(os.path.join(test_set_dir, "payments_log.csv"))
    transactions_df = pd.read_csv(os.path.join(test_set_dir, "transactions_log.csv"))

    # Load the pre-trained model
    model_path = 'rg_multi_class_mode_cat_fin.joblib'  # Adjust as needed
    rf_model = joblib.load(model_path)

    # Feature engineering (assumes a helper script handles this transformation)
    from feature_engineering import process_test_data
    test_features = process_test_data(account_state_df, transactions_df, payments_df)

    # Predict on test data
    test_features['predicted_class'] = rf_model.predict(test_features.drop(['agent_id', 'week'], axis=1))

    # Apply cascading logic
    results_with_cascading_logic = enforce_cascading_weekly_logic(test_features)

    # Aggregate final predictions
    final_results = aggregate_final_predictions(results_with_cascading_logic)

    # Save results to CSV
    output_file = os.path.join(results_dir, "results.csv")
    final_results.to_csv(output_file, index=False)
    print(f"Final results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_set_dir",
        type=str,
        required=True,
        help="Directory containing the test set files (account_state_log.csv, payments_log.csv, transactions_log.csv)"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory where the results.csv file will be saved"
    )
    args = parser.parse_args()
    main(args.test_set_dir, args.results_dir)