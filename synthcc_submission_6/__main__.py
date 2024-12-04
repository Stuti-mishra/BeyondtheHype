import argparse
import os
import sklearn
from sklearn.model_selection import train_test_split# Create a multi-class target
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PREDICTION_WINDOW_MONTHS = [3, 6, 9, 12]  # Constant for this charge-off prediction task.


def main(test_set_dir: str, results_dir: str):
    
    # Load test set data.
    # account_state_log = pd.read_csv(os.path.join(test_set_dir, "account_state_log.csv"))
    # payments_log = pd.read_csv(os.path.join(test_set_dir, "payments_log.csv"))
    # transactions_log = pd.read_csv(os.path.join(test_set_dir, "transactions_log.csv"))

    # ---------------------------------
    account_state_log = pd.read_csv('../synthcc_train_set/account_state_log.csv')
    transactions_log = pd.read_csv('../synthcc_train_set/transactions_log.csv', parse_dates=['timestamp'])
    payments_log = pd.read_csv('../synthcc_train_set/payments_log.csv', parse_dates=['timestamp'])

    account_state_log['timestamp'] = pd.to_datetime(account_state_log['timestamp']).dt.date
    payments_log['timestamp'] = pd.to_datetime(payments_log['timestamp']).dt.date
    transactions_log['timestamp'] = pd.to_datetime(transactions_log['timestamp']).dt.date

# Sort all dataframes by timestamp
    account_state_log = account_state_log.sort_values(['agent_id', 'timestamp'])
    payments_log = payments_log.sort_values(['agent_id', 'timestamp'])
    transactions_log = transactions_log.sort_values(['agent_id', 'timestamp'])

    # Create transaction aggregations by date and agent
    transaction_features = (
        transactions_log
        .groupby(['agent_id', 'timestamp', 'merchant_category'])
        .agg({
            'amount': ['count', 'sum']  # Count of transactions and sum of amounts per category
        })
        .reset_index()
    )
    transaction_features.columns = ['agent_id', 'timestamp', 'merchant_category', 
                                        'category_transaction_count', 'category_amount_sum']


        # Pivot the merchant categories to create separate columns
    category_features = (
            transaction_features
            .pivot_table(
                index=['agent_id', 'timestamp'],
                columns='merchant_category',
                values=['category_transaction_count', 'category_amount_sum'],
                fill_value=0
            )
            .reset_index()
        )

        # Flatten column names
    category_features.columns = [
            f"{col[0]}_{col[1]}".replace(' ', '_').lower() 
            if isinstance(col, tuple) else col 
            for col in category_features.columns
        ]

        # Create status aggregations
    status_features = (
            transactions_log
            .groupby(['agent_id', 'timestamp'])
            .agg({
                'status': lambda x: (x == 'approved').sum(),  # Count of approved
            })
            .rename(columns={'status': 'approved_count'})
            .reset_index()
        )


        # Add declined count
    status_features['declined_count'] = (
            transactions_log
            .groupby(['agent_id', 'timestamp'])
            .size()
            .reset_index(name='total_count')
            ['total_count'] - status_features['approved_count']
        )

    daily_payments = payments_log.groupby(['agent_id', 'timestamp'])['amount'].sum().reset_index()


    labels_log = pd.read_csv("../synthcc_train_set/labels.csv")
    category_features = category_features.rename(
            columns={
                'agent_id_': 'agent_id',
                'timestamp_': 'timestamp'
            }
        )
        # Merge daily_payments with category_features on agent_id and timestamp
    merged_data = pd.merge(
            status_features, 
            category_features, 
            on=['agent_id', 'timestamp'], 
            how='inner'
        )

        # Perform a left join with status_features
    merged_data = pd.merge(
            merged_data, 
            daily_payments, 
            on=['agent_id', 'timestamp'], 
            how='left'
        )

        # Fill missing values in status_features with 0
    merged_data['approved_count'] = merged_data['approved_count'].fillna(0).astype(int)
    merged_data['declined_count'] = merged_data['declined_count'].fillna(0).astype(int)
    merged_data['amount'] = merged_data['amount'].fillna(0).astype(int)
    merged_data.columns
    merged_data= merged_data.rename(
        columns={
            'amount':'daily_spent'
        }
    )
    account_state_log['timestamp'] = pd.to_datetime(account_state_log['timestamp']).dt.date
    payments_log['timestamp'] = pd.to_datetime(payments_log['timestamp']).dt.date

    # Sort all dataframes by timestamp
    account_state_log = account_state_log.sort_values(['agent_id', 'timestamp'])
    payments_log = payments_log.sort_values(['agent_id', 'timestamp'])

    # Resample credit data to daily frequency and forward fill
    credit_features = account_state_log.pivot_table(
        index=['timestamp', 'agent_id'],
        values=['credit_balance', 'credit_utilization', 'interest_rate', 
                'min_payment_factor', 'current_missed_payments']
    ).reset_index()

    # Calculate daily payment amounts
    daily_payments = payments_log.groupby(['agent_id', 'timestamp'])['amount'].sum().reset_index()

    # Merge credit features with payments
    features = pd.merge(account_state_log, daily_payments, 
                        on=['agent_id', 'timestamp'], how='left')
    features['amount'] = features['amount'].fillna(0)
    # Convert dates to datetime if needed
    features['timestamp'] = pd.to_datetime(features['timestamp'])
    merged_data['timestamp'] = pd.to_datetime(merged_data['timestamp'])

    # Merge the dataframes
    result = pd.merge(
        features,
        merged_data,
        on=['agent_id', 'timestamp'],
        how='inner',
        suffixes=('_features', '_merged')
    )

    print(f"Original Features Shape: {features.shape}")
    print(f"Original Merged Data Shape: {merged_data.shape}")
    print(f"Final Merged Shape: {result.shape}")

    merged_data = pd.merge(result, labels_log, on='agent_id', how='left')
    aggregated_data=merged_data
    from sklearn.model_selection import train_test_split# Create a multi-class target
    def create_multiclass_target(row):
        if row['charge_off_within_3_months'] == 1:
            return 1
        elif row['charge_off_within_6_months'] == 1:
            return 2
        elif row['charge_off_within_9_months'] == 1:
            return 3
        elif row['charge_off_within_12_months'] == 1:
            return 4
        else:
            return 0

    aggregated_data['multi_class_target'] = aggregated_data.apply(create_multiclass_target, axis=1)

    # Include agent_id as a separate column in X
    X = aggregated_data[high_importance_features]
    y = aggregated_data['multi_class_target']

    # Train/test split, keeping agent_id for reference
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Separate agent_id from training data
    X_train_features = X_train.drop(columns=['agent_id'])
    X_test_features = X_test.drop(columns=['agent_id'])

    # Train a multi-class classifier
    from xgboost import XGBClassifier
    model = XGBClassifier(objective='multi:softmax', num_class=5, eval_metric='mlogloss')
    model.fit(X_train_features, y_train)

    # Evaluate
    y_pred = model.predict(X_test_features)

    # Attach agent_id back to results for reference
    results_df = X_test.copy()
    results_df['true_target'] = y_test.values
    results_df['predicted_target'] = y_pred

    # Classification report
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))

# Save results for reference
    results_file_path = "results_predictions_with_agent_id.csv"
    results_df.to_csv(results_file_path, index=False)
    # Create a new instance of the model
    loaded_model = xgb.XGBClassifier()  # Or XGBRegressor

    # Load the model
    loaded_model.load_model('xgboost_model.json')
    # Make predictions
    predictions = loaded_model.predict(X)  
    grouped = results_df.groupby(['agent_id', 'predicted_target']).size().reset_index(name='count')
    grouped = results_df.groupby(['agent_id', 'predicted_target']).size().reset_index(name='count')

# For each 'agent_id', find the 'predicted_target' with the maximum count
    majority_label = grouped.loc[grouped.groupby('agent_id')['count'].idxmax()]

    # Display the result
    print(majority_label)
    unique_predicted_targets = majority_label['predicted_target'].unique()

    print("Unique entries in 'predicted_target':", unique_predicted_targets)

# For each 'agent_id', find the 'predicted_target' with the maximum count
    majority_label = grouped.loc[grouped.groupby('agent_id')['count'].idxmax()]

# Display the result
    print(majority_label)
    mapping = {
    0: [0, 0, 0, 0],
    1: [1, 1, 1, 1],
    2: [0, 1, 1, 1],
    3: [0, 0, 1, 1],
    4: [0, 0, 0, 1]
}

# Adding the new columns
columns = [
    'charge_off_within_3_months',
    'charge_off_within_6_months',
    'charge_off_within_9_months',
    'charge_off_within_12_months'
]

for i, col in enumerate(columns):
    majority_label[col] = majority_label['predicted_target'].map(lambda x: mapping[x][i])

majority_label.drop(columns=['predicted_target', 'count'])

# Make predictions
# The resulting dataframe `merged_data` now contains all the desired information
    # START PROCESSING TEST SET INPUTS
    # Beep boop bop you should do something with test inputs unlike this script.

    # In lieu of doing something test inputs, maybe you "learned" from training data that
    #  30% of accounts are charge-off across all periods (not true), so you randomly 
    #  guess with that percentage.
    


    # END PROCESSING TEST SET INPUTS
    # ---------------------------------

    # NOTE: name "results.csv" is a must.
    predictions.to_csv(os.path.join(results_dir, "results.csv"), index=False)

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
