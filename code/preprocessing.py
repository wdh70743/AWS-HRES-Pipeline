import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

if __name__ == '__main__':
    # Load the collected data
    input_path = '/opt/ml/processing/input/collected_data.csv'
    df = pd.read_csv(input_path)

    # Preprocessing steps
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m')
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month

    df_pivot = df.pivot_table(
        index=['latitude', 'longitude', 'datetime', 'year', 'month', 'cluster'],
        columns='parameter',
        values='value'
    ).reset_index()

    df = df_pivot.sort_values(by=['cluster', 'year', 'month'])
    df['ALLSKY_SFC_SW_DWN_shifted'] = df.groupby(['latitude', 'longitude'])['ALLSKY_SFC_SW_DWN'].shift(1)
    df.loc[df.groupby(['latitude', 'longitude'])['datetime'].idxmax(), 'ALLSKY_SFC_SW_DWN_shifted'] = pd.NA
    df_cleaned = df.dropna(subset=['ALLSKY_SFC_SW_DWN_shifted'])

    # Prepare features and target
    X = df_cleaned.drop(['datetime', 'cluster', 'CLRSKY_SFC_SW_DWN', 'ALLSKY_SFC_SW_DWN', 'ALLSKY_SFC_SW_DWN_shifted', 'WS50M','ALLSKY_KT'], axis=1)
    Y = df_cleaned['ALLSKY_SFC_SW_DWN_shifted']

    # Split the data
    X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Save the preprocessed data
    train_output = pd.concat([X_train, y_train], axis=1)
    val_output = pd.concat([X_val, y_val], axis=1)
    test_output = pd.concat([X_test, y_test], axis=1)

    train_output.to_csv('/opt/ml/processing/train/train.csv', index=False)
    val_output.to_csv('/opt/ml/processing/validation/validation.csv', index=False)
    test_output.to_csv('/opt/ml/processing/test/test.csv', index=False)

    print("Preprocessing completed. Data split into train, validation, and test sets.")