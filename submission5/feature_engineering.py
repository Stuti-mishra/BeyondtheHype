import pandas as pd

def aggregate_weekly(df, date_col, group_col, agg_dict):
    """
    Aggregates data weekly based on the specified aggregation dictionary.
    """
    df['week'] = df[date_col].dt.to_period('W').apply(lambda r: r.start_time)
    return df.groupby([group_col, 'week']).agg(agg_dict).reset_index()

def process_test_data(account_state_df, transactions_df, payments_df):
    """
    Processes test data for predictions, including weekly aggregation and merging features.
    """
    # Weekly aggregation for account state
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

    # Weekly aggregation for transactions
    transaction_features = aggregate_weekly(
        transactions_df, 'timestamp', 'agent_id',
        {
            'amount': ['mean', 'sum', 'max', 'std'],
            'online': ['sum'],
            'status': lambda x: (x == 'approved').sum() / len(x)  # Approval rate
        }
    )
    transaction_features.columns = ['agent_id', 'week'] + [f"transaction_{col[0]}_{col[1]}" for col in transaction_features.columns[2:]]

    # Weekly aggregation for payments
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

    # Forward-fill missing weeks for each agent
    features = features.groupby('agent_id').apply(lambda x: x.sort_values('week').ffill().bfill()).reset_index(drop=True)

    return features