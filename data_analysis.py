import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from simulation import create_dataset

df = create_dataset(n_samples=14000, save_to_csv=True)

train_val, test = train_test_split(df, test_size=0.2, random_state=42)
train, val = train_test_split(train_val, test_size=0.2, random_state=42)

print("Dataset sizes:")
print(f"Training samples: {len(train)}")
print(f"Validation samples: {len(val)}")
print(f"Test samples: {len(test)}")

print("\nClass distribution across splits:")
print("\nTraining set:")
print(train['wettability_class'].value_counts(normalize=True))
print("\nValidation set:")
print(val['wettability_class'].value_counts(normalize=True))
print("\nTest set:")
print(test['wettability_class'].value_counts(normalize=True))

numerical_features = ['contact_angle', 'surface_energy', 'roughness_ra', 'porosity', 
                     'thermal_conductivity', 'youngs_modulus', 'density']

print("\nSummary statistics for numerical features:")
print(train[numerical_features].describe())

correlations = train[numerical_features].corr()
print("\nFeature correlations:")
print(correlations)

print("\nContact angle statistics by material:")
print(train.groupby('material')['contact_angle'].describe())

print("\nContact angle statistics by surface treatment:")
print(train.groupby('surface_treatment')['contact_angle'].describe())

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

categorical_features = ['material', 'surface_treatment']
X = train[numerical_features + categorical_features].copy()

for feat in categorical_features:
    le = LabelEncoder()
    X[feat] = le.fit_transform(X[feat])


le_target = LabelEncoder()
y = le_target.fit_transform(train['wettability_class'])

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature importance:")
print(importance_df)


analysis_summary = {
    'dataset_sizes': {
        'train': len(train),
        'validation': len(val),
        'test': len(test)
    },
    'class_distribution': train['wettability_class'].value_counts().to_dict(),
    'feature_importance': importance_df.to_dict(),
    'correlations': correlations.to_dict(),
    'summary_statistics': train[numerical_features].describe().to_dict()
}

import json
with open('wettability_analysis_summary.json', 'w') as f:
    json.dump(analysis_summary, f, indent=4)

print("\nAnalysis summary has been saved to 'wettability_analysis_summary.json'")