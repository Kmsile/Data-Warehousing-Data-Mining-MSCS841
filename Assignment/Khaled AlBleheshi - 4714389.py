##Coded by Khaled AlBleheshi - 4714389
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os


## Importing the dataset
df = sns.load_dataset('titanic')
df_clean = df.copy()

## Dataset Overview
numeric_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(include=['object', 'category', 'string']).columns
print("-" * 50)
print("DATASET OVERVIEW")
print("-" * 50)
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
print(f"\nColumn names and types:")
print(df.dtypes)
print(f"\nFirst 5 rows:")
print(df.head(5))
print("\n" + "-" * 50)
print("NUMERIC ATTRIBUTES - SUMMARY STATISTICS")
print("-" * 50)
print(df[numeric_cols].describe())
print("\n" + "-" * 50)
print("CATEGORICAL ATTRIBUTES - FREQUENCY TABLES")
print("-" * 50)
for col in categorical_cols:
    print(f"\n--- {col} ---")
    print(df[col].value_counts().head())
    
## Data Quality Issues Detection
print("\n" + "-" * 50)
print("DATA QUALITY ISSUES DETECTED")
print("-" * 50)

## Missing Values
print("\n--- Missing Values ---")
missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_percent})
print(missing_df[missing_df['Missing Count'] > 0])

## Duplicate Rows
duplicates = df.duplicated().sum()
print("\n--- Duplicate Rows ---")
print(f"Number of duplicate rows: {duplicates}")

## Outliers
print(f"\n--- Outliers in Fare ---")
Q1 = df['fare'].quantile(0.25)
Q3 = df['fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['fare'] < lower_bound) | (df['fare'] > upper_bound)]
print(f"Number of outlier rows: {len(outliers)}")
print(f"Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")

## Data Preprocessing Steps
print("\n" + "-" * 50)
print("DATA PREPROCESSING STEPS")
print("-" * 50)

## Handling Missing Values
print("\n--- A. Handling Missing Values ---")
age_median = df_clean['age'].median()
df_clean['age'] = df_clean['age'].fillna(age_median)
embarked_mode = df_clean['embarked'].mode()[0]
df_clean['embarked'] = df_clean['embarked'].fillna(embarked_mode)
df_clean = df_clean.drop(columns=['deck'])
print(f"Filled 'embarked' missing values with mode: '{embarked_mode}'")
print(f"Filled 'age' missing values with median: {age_median:.1f}")
print("Dropped 'deck' column (77% missing, unreliable to impute)")

## Handling Outliers
print("\n--- B. Handling Outliers ---")
df_clean['fare'] = np.where(df_clean['fare'] > upper_bound, upper_bound, df_clean['fare'])
print(f"Capped 'fare' outliers at upper bound: {upper_bound:.2f}")

## Removing Duplicates
print("\n--- C. Removing Duplicates ---")
before_rows = len(df_clean)
df_clean = df_clean.drop_duplicates()
after_rows = len(df_clean)
print(f"Removed {before_rows - after_rows} duplicate rows")

## Data Type Fixes
print("\n--- D. Data Type Fixes ---")
df_clean['pclass'] = df_clean['pclass'].astype('category')
print("Converted 'pclass' to category type")

## Noisy Data
print("\n--- Noisy Data ---")
print("No explicit noise detected, but fare=$0 for 15 passengers may indicate crew/free tickets")
print("Age precision varies (decimals for infants < 1 year)")

## Inconsistent Formats
print("\n--- Inconsistent Formats ---")
print("Age column contains decimal values (0.42, 0.67, etc.) for infants - biologically valid")
print("Embarked uses single-letter codes (S/C/Q) while Embark_Town uses full names")

## Incorrect Data Entries
print("\n--- Incorrect Data Entries ---")
print("Fare=$0: 15 passengers - likely crew or complimentary tickets")
print("Age=0.42: Valid infant age (5 months)")

## Summary
print("\n" + "-" * 50)
print("DATASET SUMMARY")
print("-" * 50)
print(f"Original rows: {df.shape[0]} | Cleaned rows: {df_clean.shape[0]}")
print(f"Original columns: {df.shape[1]} | Cleaned columns: {df_clean.shape[1]}")

## Before and After Comparison
print("\n--- Before/After Comparison ---")
comparison = pd.DataFrame({
    'Metric': ['Missing Values (age)', 'Missing Values (embarked)', 'Missing Values (deck)', 'Duplicates'],
    'Before': [df['age'].isnull().sum(), df['embarked'].isnull().sum(),df['deck'].isnull().sum(), df.duplicated().sum()],
    'After': [df_clean['age'].isnull().sum(), df_clean['embarked'].isnull().sum(),'Column Removed', df_clean.duplicated().sum()]})
print(comparison.to_string(index=False))

## Save to Desktop
home = os.path.expanduser('~')
possible_paths = [os.path.join(home, 'OneDrive', 'Desktop'),os.path.join(home, 'Desktop'),]
desktop = None
for path in possible_paths:
    if os.path.exists(path):
        desktop = path
        break
if desktop is None:
    desktop = home
output_file = os.path.join(desktop, 'titanic_cleaned.csv')
df_clean.to_csv(output_file, index=False)
print(f"\nCleaned dataset saved to: {output_file}")

## Figures Overview
plt.figure(figsize=(15, 10))
plt.subplot(1, 3, 1)
sns.histplot(df['age'].dropna(), bins=20, kde=True, color='skyblue')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.subplot(1, 3, 2)
sns.boxplot(y=df['fare'], color='lightcoral')
plt.title('Boxplot of Fare (Outliers Detection)')
plt.subplot(1, 3, 3)
sns.countplot(x='sex', data=df, hue='sex', palette='viridis', legend=False)
plt.title('Gender Distribution')
plt.tight_layout()
plt.show()