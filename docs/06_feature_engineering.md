# ⚙️ Feature Engineering Guide

> Transform raw data into meaningful features that improve analysis and model performance

---

## What is Feature Engineering?

Feature engineering is the process of using domain knowledge to **create new variables** from existing ones that better represent the problem.

```
Raw column: ApplicantIncome = 5000, LoanAmount = 2500
New feature: Debt_to_Income = 2500 / 5000 = 0.50  ← Much more informative!
```

---

## Why It Matters

- Better features → Better models
- Domain knowledge encoded into data
- Reveals patterns not visible in raw data
- Often more impactful than algorithm choice

---

## Types of Features

```
1. MATHEMATICAL     → Ratios, differences, products
2. CATEGORICAL      → Encode text as numbers
3. DATE/TIME        → Extract year, month, day, weekday
4. AGGREGATIONS     → Group statistics
5. BINNING          → Convert numbers to categories
6. INTERACTION      → Combine two columns
7. FLAGS            → Binary indicators (0/1)
8. TEXT             → Extract from strings
```

---

## 1. Mathematical Features

```python
import pandas as pd
import numpy as np

# ===== RATIOS =====
# Most powerful for financial analysis
df['debt_to_income'] = df['loan_amount'] / df['income']
df['expense_ratio'] = df['expenses'] / df['revenue']
df['growth_rate'] = (df['current'] - df['previous']) / df['previous']

# ===== DIFFERENCES =====
df['income_gap'] = df['applicant_income'] - df['coapplicant_income']
df['budget_variance'] = df['actual'] - df['budgeted']
df['days_since_last'] = (pd.Timestamp.now() - df['last_date']).dt.days

# ===== PRODUCTS =====
df['total_income'] = df['applicant_income'] + df['coapplicant_income']
df['monthly_payment'] = df['loan_amount'] / df['term_months']
df['annual_revenue'] = df['monthly_revenue'] * 12

# ===== LOG TRANSFORMATION =====
# Use for skewed distributions
df['income_log'] = np.log1p(df['income'])  # log(1 + x) handles zeros
df['amount_log'] = np.log(df['amount'].clip(lower=1))  # clip avoids log(0)

# ===== SQUARED / POWER =====
# Capture non-linear relationships
df['age_squared'] = df['age'] ** 2
df['income_sqrt'] = np.sqrt(df['income'])

# Real Example: Loan Analysis
df_loans = df.copy()

df_loans['total_income'] = df_loans['ApplicantIncome'] + df_loans['CoapplicantIncome']
df_loans['debt_to_income'] = df_loans['LoanAmount'] / df_loans['ApplicantIncome']
df_loans['monthly_payment'] = df_loans['LoanAmount'] / (df_loans['Loan_Amount_Term'] / 12)
df_loans['income_per_dependent'] = df_loans['total_income'] / (df_loans['Dependents_numeric'] + 1)

print("✅ Mathematical features created:")
print(df_loans[['total_income', 'debt_to_income', 'monthly_payment']].head())
```

---

## 2. Categorical Encoding

```python
# ===== ONE-HOT ENCODING =====
# Best for: Nominal categories (no order)
# Example: Urban/Rural/Semiurban → 3 binary columns

df_encoded = pd.get_dummies(df, columns=['Property_Area'], drop_first=True)
# drop_first=True avoids multicollinearity

# Manual one-hot encoding
df['is_urban'] = (df['Property_Area'] == 'Urban').astype(int)
df['is_rural'] = (df['Property_Area'] == 'Rural').astype(int)

# ===== LABEL ENCODING =====
# Best for: Ordinal categories (has order)
# Example: Low/Medium/High → 0/1/2

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['risk_encoded'] = le.fit_transform(df['risk_category'])

# Manual label encoding
risk_map = {'Low': 0, 'Medium': 1, 'High': 2}
df['risk_encoded'] = df['risk_category'].map(risk_map)

# ===== BINARY ENCODING =====
# Best for: Yes/No, True/False

df['is_married'] = (df['Married'] == 'Yes').astype(int)
df['is_graduate'] = (df['Education'] == 'Graduate').astype(int)
df['is_self_employed'] = (df['Self_Employed'] == 'Yes').astype(int)
df['has_credit_history'] = df['Credit_History'].astype(int)

# ===== FREQUENCY ENCODING =====
# Best for: High cardinality categories
# Replaces category with its frequency in dataset

freq_map = df['city'].value_counts(normalize=True)
df['city_frequency'] = df['city'].map(freq_map)

# ===== TARGET ENCODING =====
# Best for: When category correlates with target
# Replaces category with mean of target

target_mean = df.groupby('city')['target'].mean()
df['city_target_encoded'] = df['city'].map(target_mean)
```

---

## 3. Date/Time Features

```python
# Convert to datetime first
df['date'] = pd.to_datetime(df['date'])

# ===== EXTRACT DATE PARTS =====
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek   # 0=Monday, 6=Sunday
df['day_name'] = df['date'].dt.day_name()     # 'Monday', 'Tuesday'...
df['quarter'] = df['date'].dt.quarter
df['week_number'] = df['date'].dt.isocalendar().week

# ===== BINARY TIME FLAGS =====
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)

# ===== TIME DIFFERENCES =====
reference_date = pd.Timestamp('2024-01-01')
df['days_since_reference'] = (df['date'] - reference_date).dt.days

df['days_since_last_purchase'] = (
    pd.Timestamp.now() - df['last_purchase_date']
).dt.days

# ===== CYCLICAL ENCODING =====
# Month 12 and month 1 are close → use sin/cos

df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
```

---

## 4. Aggregation Features

```python
# ===== GROUP STATISTICS =====
# Add group-level statistics to each row

# Mean by group
df['avg_income_by_area'] = df.groupby('Property_Area')['income'].transform('mean')

# Median by group
df['median_loan_by_edu'] = df.groupby('Education')['LoanAmount'].transform('median')

# Count by group
df['customers_per_city'] = df.groupby('city')['customer_id'].transform('count')

# Sum by group
df['total_purchases_per_customer'] = df.groupby('customer_id')['amount'].transform('sum')

# Multiple aggregations
agg_features = df.groupby('customer_id').agg(
    total_amount=('amount', 'sum'),
    avg_amount=('amount', 'mean'),
    transaction_count=('transaction_id', 'count'),
    max_amount=('amount', 'max'),
    days_active=('date', lambda x: (x.max() - x.min()).days)
).reset_index()

df = df.merge(agg_features, on='customer_id', how='left')

# ===== ROLLING FEATURES (Time Series) =====
df = df.sort_values('date')

# Rolling average (last 7 days)
df['rolling_7d_avg'] = df['sales'].rolling(window=7).mean()

# Rolling sum (last 30 days)
df['rolling_30d_sum'] = df['sales'].rolling(window=30).sum()

# Lag features (previous values)
df['sales_yesterday'] = df['sales'].shift(1)
df['sales_last_week'] = df['sales'].shift(7)
```

---

## 5. Binning - Numbers to Categories

```python
# ===== EQUAL WIDTH BINS =====
# Each bin covers same range

df['age_group'] = pd.cut(
    df['age'],
    bins=[0, 25, 35, 50, 65, 100],
    labels=['Gen Z', 'Millennial', 'Gen X', 'Boomer', 'Senior']
)

# ===== EQUAL FREQUENCY BINS (Quantiles) =====
# Each bin has same number of records

df['income_quartile'] = pd.qcut(
    df['income'],
    q=4,
    labels=['Q1-Low', 'Q2-Mid-Low', 'Q3-Mid-High', 'Q4-High']
)

# Simple quartiles (numeric)
df['income_quartile_num'] = pd.qcut(df['income'], q=4, labels=[1, 2, 3, 4])

# ===== CUSTOM BINS =====
# Based on domain knowledge

def risk_category(dti):
    """Categorize Debt-to-Income ratio"""
    if dti < 0.2:
        return 'Very Low Risk'
    elif dti < 0.4:
        return 'Low Risk'
    elif dti < 0.6:
        return 'Medium Risk'
    elif dti < 0.8:
        return 'High Risk'
    else:
        return 'Very High Risk'

df['risk_category'] = df['debt_to_income'].apply(risk_category)

# Using case_when style (pd.cut with custom bins)
df['credit_tier'] = pd.cut(
    df['credit_score'],
    bins=[0, 579, 669, 739, 799, 850],
    labels=['Poor', 'Fair', 'Good', 'Very Good', 'Exceptional']
)
```

---

## 6. Interaction Features

```python
# ===== MULTIPLY TWO FEATURES =====
# Capture combined effect

df['income_x_credit'] = df['income'] * df['credit_score']
df['amount_x_term'] = df['loan_amount'] * df['term_months']

# ===== COMBINE CATEGORICAL =====
df['area_education'] = df['Property_Area'] + '_' + df['Education']
# Creates: 'Urban_Graduate', 'Rural_Not Graduate', etc.

# ===== RATIO OF RATIO =====
df['adjusted_dti'] = df['debt_to_income'] / df['credit_score_normalized']
```

---

## 7. Flag Features (Binary Indicators)

```python
# ===== THRESHOLD FLAGS =====
df['high_income'] = (df['income'] > df['income'].quantile(0.75)).astype(int)
df['large_loan'] = (df['loan_amount'] > 200).astype(int)
df['long_term'] = (df['term_months'] > 360).astype(int)

# ===== MULTIPLE CONDITIONS =====
df['ideal_candidate'] = (
    (df['credit_history'] == 1) &
    (df['debt_to_income'] < 0.4) &
    (df['income'] > df['income'].median())
).astype(int)

# ===== NULL FLAGS =====
# Keep info about where nulls were
df['income_was_null'] = df['income'].isnull().astype(int)
df['credit_was_null'] = df['credit_score'].isnull().astype(int)

# ===== ANOMALY FLAGS =====
from scipy import stats

z_scores = np.abs(stats.zscore(df['income'].dropna()))
df['income_anomaly'] = (z_scores > 3).astype(int)
```

---

## Complete Feature Engineering Pipeline

```python
def engineer_features(df):
    """
    Complete feature engineering pipeline
    
    Parameters:
    -----------
    df : pd.DataFrame - Cleaned dataset
    
    Returns:
    --------
    pd.DataFrame - Dataset with engineered features
    """
    
    df_fe = df.copy()
    features_created = []
    
    print("=" * 60)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 60)
    
    # 1. Mathematical features (if columns exist)
    if 'LoanAmount' in df.columns and 'ApplicantIncome' in df.columns:
        df_fe['debt_to_income'] = df_fe['LoanAmount'] / df_fe['ApplicantIncome']
        df_fe['total_income'] = df_fe['ApplicantIncome'] + df_fe.get('CoapplicantIncome', 0)
        features_created.extend(['debt_to_income', 'total_income'])
    
    # 2. Binary encoding
    binary_map = {
        'Gender': ('Male', 'is_male'),
        'Married': ('Yes', 'is_married'),
        'Education': ('Graduate', 'is_graduate'),
        'Self_Employed': ('Yes', 'is_self_employed')
    }
    
    for col, (positive_val, new_col) in binary_map.items():
        if col in df_fe.columns:
            df_fe[new_col] = (df_fe[col] == positive_val).astype(int)
            features_created.append(new_col)
    
    # 3. Numeric dependents
    if 'Dependents' in df_fe.columns:
        df_fe['dependents_numeric'] = df_fe['Dependents'].replace('3+', '3').astype(int)
        features_created.append('dependents_numeric')
    
    # 4. Risk score
    if all(c in df_fe.columns for c in ['Credit_History', 'debt_to_income']):
        df_fe['risk_score'] = (
            (df_fe['Credit_History'] * 40) +
            ((1 - df_fe['debt_to_income'].clip(0, 1)) * 30) +
            ((df_fe.get('total_income', df_fe['ApplicantIncome']) / 
              df_fe.get('total_income', df_fe['ApplicantIncome']).max()) * 30)
        )
        df_fe['risk_category'] = pd.cut(
            df_fe['risk_score'],
            bins=[0, 40, 70, 100],
            labels=['High', 'Medium', 'Low']
        )
        features_created.extend(['risk_score', 'risk_category'])
    
    print(f"\n✅ Features created: {len(features_created)}")
    for feat in features_created:
        print(f"   • {feat}")
    
    print(f"\nFinal shape: {df_fe.shape}")
    
    return df_fe


# Usage
df_engineered = engineer_features(df_cleaned)
```

---

## Feature Engineering Checklist

Before modeling, verify your features:

- [ ] No infinite values (`df.replace([np.inf, -np.inf], np.nan)`)
- [ ] No nulls introduced by engineering
- [ ] Ratios handle division by zero
- [ ] Encoded all categorical variables
- [ ] Datetime columns extracted
- [ ] Feature names are descriptive
- [ ] No data leakage (future info in features)

---

## Quick Reference

| Goal | Technique | Example |
|------|-----------|---------|
| Capture ratio | Division | `loan / income` |
| Handle skew | Log transform | `np.log1p(income)` |
| Encode Yes/No | Binary | `(col == 'Yes').astype(int)` |
| Encode categories | One-hot | `pd.get_dummies()` |
| Number to group | Cut | `pd.cut(age, bins=[0,25,50,100])` |
| Group statistics | GroupBy | `df.groupby().transform('mean')` |
| Time patterns | dt accessor | `df['date'].dt.month` |
| Anomaly flag | Z-score | `abs(zscore) > 3` |

---

## Next Steps

→ [Visualization Guide](07_visualization.md)
