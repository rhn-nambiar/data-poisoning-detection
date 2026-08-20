import pandas as pd
import random
import string
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

# Load the data
data_path = "data/poisoned_combined4.csv"
data = pd.read_csv(data_path)

data_path1 = "data/valid_license_plates.csv"
data1 = pd.read_csv(data_path1)
if 'License Plate' in data1.columns:
    data1.rename(columns={'License Plate': 'LicensePlate'}, inplace=True)
data1['LicensePlate'] = data1['LicensePlate'].str.replace(" ", "", regex=True)

# Extract details from license plates
data['state'] = data['LicensePlate'].str[:2]
data['series'] = data['LicensePlate'].str[4:6]
data['rto'] = data['LicensePlate'].str[2:4]
data['lnum'] = data['LicensePlate'].str[6:]
data['lnum'] = data['lnum'].astype(str).fillna('')  # Ensure 'lnum' column is clean
data['rto'] = data['rto'].apply(lambda x: x if str(x).isdigit() else '0')

# Rule-based anomaly detection
def rule_based_detection(data):
    valid_rto_codes = {
        "AP": range(1, 40), "AR": range(1, 14), "AS": range(1, 38),
       "BR": range(1, 45), "CG": range(1, 28), "GA": range(1, 13),
    "GJ": range(1, 39), "HR": range(1, 75), "HP": range(1, 100),
    "JH": range(1, 23), "KA": range(1, 100), "KL": range(1, 100),
    "MP": range(1, 72), "MH": range(1, 61), "MN": range(1, 8),
    "ML": range(1, 11), "MZ": range(1, 6), "NL": range(1, 12),
    "OD": range(1, 36), "PB": range(1, 66), "RJ": range(1, 53),
    "SK": range(1, 9), "TN": range(1, 100), "TS": range(1, 37),
    "TR": range(1, 9), "UP": range(1, 97), "UK": range(1, 21),
    "WB": range(1, 100), "AN": range(1, 4), "CH": range(1, 5),
    "DD": range(1, 5), "DL": range(1, 14), "JK": range(1, 23),
    "LA": range(1, 3), "LD": range(1, 2), "PY": range(1, 6)
    }
    invalid1 = ['GA', 'GB', 'CD', 'CC', 'EV']
    invalid2 = ['I', 'O', 'Q']
    anom1 = data[~data['state'].isin(valid_rto_codes.keys())]

    anom2 = data[data['series'].isin(invalid1)]

    anom3 = data[(data['series'].fillna('').str[0].isin(invalid2)) | 
             (data['series'].fillna('').str[1].isin(invalid2))]

    anom4 = data[(data['series'].fillna('').str[0].str.isdigit()) | 
             (data['series'].fillna('').str[1].str.isdigit())]
    data['lnum'] = data['lnum'].astype(str).fillna('')

    invalid_numbers = data[~data['lnum'].str.match(r'^\d+$')]

    anom5 = data[data.apply(lambda row: int(row['rto']) not in valid_rto_codes.get(row['state'], []) if row['rto'].isdigit() and row['state'] in valid_rto_codes else True, axis=1)]

    anom6 = data[~data['lnum'].str.match(r'^\d+$')]
    

    anom_all = pd.concat([anom1, anom2, anom3, anom4, anom5, anom6]).drop_duplicates()
    return anom_all


rule_based_anomalies = rule_based_detection(data)

# Remove rule-based anomalies
optimized_data = data[~data['LicensePlate'].isin(rule_based_anomalies['LicensePlate'])]

# Encode columns
def encode_categorical_columns(df, columns):
    encoders = {}
    df = df.copy()
    for col in columns:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].fillna('missing'))
        encoders[col] = encoder
    return df, encoders

categorical_columns = ['state', 'rto', 'series']
optimized_data, encoders = encode_categorical_columns(optimized_data, categorical_columns)

numerical_data = optimized_data.drop(columns=['LicensePlate']).fillna(0)  
numerical_data = numerical_data.select_dtypes(include=[np.number]) 
numerical_data = numerical_data.replace([np.inf, -np.inf], np.nan)  
numerical_data = numerical_data.fillna(0)  
# Train Isolation Forest
isolation_forest = IsolationForest(n_estimators=3, max_samples=256, contamination='auto', random_state=42)
isolation_forest.fit(numerical_data)
optimized_data['anomaly_score'] = isolation_forest.decision_function(numerical_data)
optimized_data['isolation_flag'] = isolation_forest.predict(numerical_data)
optimized_data['isolation_flag'] = (optimized_data['isolation_flag'] == -1).astype(int)

# DBSCAN clustering
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_data)
dbscan = DBSCAN(eps=1.5, min_samples=15)
optimized_data['dbscan_label'] = dbscan.fit_predict(scaled_data)
optimized_data['clustering_flag'] = (optimized_data['dbscan_label'] == -1).astype(int)

# Combine anomaly detection results
data['rule_based_flag'] = data['LicensePlate'].isin(rule_based_anomalies['LicensePlate']).astype(int)
data['combined_score'] = (
    0.2* optimized_data['isolation_flag'] +
     0.8* optimized_data['clustering_flag'] 
    )
data['combined_score'] = data['combined_score'].fillna(0)  

# Ensure all scores are finite
data['combined_score'] = np.where(
    np.isfinite(data['combined_score']),
    data['combined_score'],
    0)
data['is_anomaly'] = (data['rule_based_flag'] == 1) | (data['combined_score'] > 0.2)

print("Total Anomalies Detected:")
print(data[data['is_anomaly'] == 1])


# Correctly label anomalies in the ground truth
data['true_anomaly'] = (~data['LicensePlate'].isin(data1['LicensePlate'])).astype(int)


# Calculate evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Get true and predicted labels
y_true = data['true_anomaly']
y_pred = data['is_anomaly']

# Compute metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='binary')
recall = recall_score(y_true, y_pred, average='binary')
f1 = f1_score(y_true, y_pred, average='binary')



# Print results
print("Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
