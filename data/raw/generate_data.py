import pandas as pd
import numpy as np
import os

# Define the number of synthetic records
N_SAMPLES = 10000
OUTPUT_PATH = 'data/raw/train.csv'

# Define the features (mimicking NYC Taxi data)
data = {}

# 1. Categorical Features
data['PULocationID'] = np.random.randint(1, 264, N_SAMPLES)
data['DOLocationID'] = np.random.randint(1, 264, N_SAMPLES)
data['VendorID'] = np.random.choice([1, 2], N_SAMPLES)

# 2. Datetime Features
start_time = pd.to_datetime('2023-01-01 00:00:00')
end_time = pd.to_datetime('2023-01-31 23:59:59')
data['tpep_pickup_datetime'] = pd.to_datetime(
    start_time + (end_time - start_time) * np.random.rand(N_SAMPLES)
)

# 3. Numeric Features
data['passenger_count'] = np.random.randint(1, 7, N_SAMPLES)
data['trip_distance'] = np.random.uniform(0.5, 20.0, N_SAMPLES).round(2)
data['fare_amount'] = np.random.uniform(5.0, 50.0, N_SAMPLES).round(2)
data['tip_amount'] = np.random.uniform(0, data['fare_amount'] * 0.3, N_SAMPLES).round(2)
data['tolls_amount'] = np.random.choice([0.0, 0.0, 0.0, 3.5, 6.25], N_SAMPLES)

# 4. Target Variable (Trip Duration in minutes)
# Duration is generally correlated with distance, plus some noise
duration_minutes = (data['trip_distance'] * np.random.uniform(3, 5, N_SAMPLES)) + np.random.normal(5, 5, N_SAMPLES)
data['duration_minutes'] = np.maximum(1.0, duration_minutes).round(1) # Ensure duration is at least 1 min

df = pd.DataFrame(data)

# Ensure datetime columns are correctly formatted
df['tpep_pickup_datetime'] = df['tpep_pickup_datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Save the DataFrame to CSV
print(f"Generating synthetic data to: {OUTPUT_PATH}")
df.to_csv(OUTPUT_PATH, index=False)
print(f"Successfully generated {len(df)} rows.")

# Display the first few rows for verification
print("\nFirst 5 rows of the generated data:")
print(df.head().to_markdown(index=False))