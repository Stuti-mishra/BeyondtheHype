import argparse
import os

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib


PREDICTION_WINDOW_MONTHS = [3, 6, 9, 12]  # Constant for this charge-off prediction task.


def main(test_set_dir: str, results_dir: str):
    
    # Load test set data.
    account_state_df = pd.read_csv(os.path.join(test_set_dir, "account_state_df.csv"))
    payments_df = pd.read_csv(os.path.join(test_set_dir, "payments_df.csv"))
    transactions_df = pd.read_csv(os.path.join(test_set_dir, "transactions_df.csv"))

    # ---------------------------------
    # START PROCESSING TEST SET INPUTS
    # Beep boop bop you should do something with test inputs unlike this script.

    # In lieu of doing something test inputs, maybe you "learned" from training data that
    #  30% of accounts are charge-off across all periods (not true), so you randomly 
    #  guess with that percentage.

    # Aggregate metrics for each agent
    aggregated_data = account_state_df.groupby('agent_id').agg({
        'credit_utilization': 'mean',
        'current_missed_payments': 'sum'
    }).reset_index()

    # Calculate average transaction and payment amounts per month and year
    transactions_df['month'] = transactions_df['timestamp'].dt.to_period('M')
    payments_df['month'] = payments_df['timestamp'].dt.to_period('M')

    avg_transaction_per_month = transactions_df.groupby(['agent_id', 'month'])['amount'].mean().groupby('agent_id').mean().reset_index(name='avg_transaction_per_month')
    avg_payment_per_month = payments_df.groupby(['agent_id', 'month'])['amount'].mean().groupby('agent_id').mean().reset_index(name='avg_payment_per_month')

    avg_transaction_per_year = transactions_df.groupby(['agent_id', transactions_df['timestamp'].dt.year])['amount'].mean().groupby('agent_id').mean().reset_index(name='avg_transaction_per_year')
    avg_payment_per_year = payments_df.groupby(['agent_id', payments_df['timestamp'].dt.year])['amount'].mean().groupby('agent_id').mean().reset_index(name='avg_payment_per_year')

    # Merge aggregated data
    aggregated_data = aggregated_data.merge(avg_transaction_per_month, on='agent_id', how='left')
    aggregated_data = aggregated_data.merge(avg_payment_per_month, on='agent_id', how='left')
    aggregated_data = aggregated_data.merge(avg_transaction_per_year, on='agent_id', how='left')
    aggregated_data = aggregated_data.merge(avg_payment_per_year, on='agent_id', how='left')
    
    agents = list(set(account_state_df.agent_id).union(set(payments_df.agent_id)).union(set(transactions_df.agent_id)))
    col_names = {months: f"charge_off_within_{months}_months" for months in PREDICTION_WINDOW_MONTHS}
    output_df = pd.DataFrame(columns=["agent_id"] + list(col_names.values()))
    output_df["agent_id"] = agents
    for months in PREDICTION_WINDOW_MONTHS:
        col_name = col_names[months]
        models = {}
        models[col_name] = joblib.load(f"rf_model_{months}.joblib")
        preds = models[col_name].predict(aggregated_data)
        # When unsure of whether their predictions span the entire set of agents to predict
        #  for, the true data scientist pads their predictions with zeros lol.
        preds = np.append(preds, [0]*(len(agents) - len(preds)))
        output_df[col_name] = preds

    # END PROCESSING TEST SET INPUTS
    # ---------------------------------

    # NOTE: name "results.csv" is a must.
    output_df.to_csv(os.path.join(results_dir, "results.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bth_test_set",
        type=str,
        required=True
    )
    parser.add_argument(
        "--bth_results",
        type=str,
        required=True
    )

    args = parser.parse_args()
    main(args.bth_test_set, args.bth_results)