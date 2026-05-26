# 🧹 Data Cleaning Guide

> Practical guide to handle the most common data quality issues in real-world datasets

---

## Why Data Cleaning Matters

> "Data scientists spend 80% of their time cleaning data and 20% complaining about it."

Raw data is almost never ready for analysis. Common issues include:
- Missing values (nulls)
- Outliers distorting results
- Duplicate records
- Inconsistent formats
- Wrong data types

---

## The 5-Step Cleaning Framework

```
1. ASSESS      → Understand what needs cleaning
2. NULLS       → Handle missing values
3. OUTLIERS    → Detect and treat extreme values
4. DUPLICATES  → Remove repeated records
5. FORMAT      → Standardize types and values
```

---

## Step 1: ASSESS - What Needs Cleaning?

```python
import pandas as pd
import numpy as np

# Always start with a full assessment
def assess_data_quality(df):
    """Quick data quality assessment"""
    
    print("=" * 60)
    print("DATA QUALITY ASSESSMENT")
    print("=" * 60)
    
    # Shape
    print(f"\n📊 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Missing values
    nulls = df.isnull().sum()
    null_pct = (nulls / len(df) * 100).round(2)
    null_summary = pd.DataFrame({
        'Missing': nulls,
        'Percentage': null_pct
    }).query('Missing > 0').sort_values('Missing', ascending=False)
    
    print(f"\n⚠️  Columns with missing values: {len(null_summary)}")
    if len(null_summary) > 0:
        print(null_summary)
    
    # Duplicates
    dups = df.duplicated().sum()
    print(f"\n🔄 Duplicate rows: {dups} ({dups/len(df)*100:.2f}%)")
    
    # Data types
    print(f"\n📋 Data types:")
    print(df.dtypes.value_counts())
    
    # Numeric summary for outlier detection
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"\n📈 Numeric columns: {list(numeric_cols)}")
    
    return null_summary

# Usage
quality_report = assess_data_quality(df)
```

---

## Step 2: NULLS - Handling Missing Values

### Strategy Decision Tree

```
Is the column important?
├── YES → What percentage is missing?
│         ├── < 5%  → Fill with mean/median/mode
│         ├── 5-30% → Fill with strategy or flag
│         └── > 30% → Consider dropping column
└── NO  → Drop column
```

### Techniques

```python
# ===== TECHNIQUE 1: Fill with Statistics =====

# Numeric columns → use median (robust to outliers)
df['amount'].fillna(df['amount'].median(), inplace=True)

# Numeric columns → use mean (when no outliers)
df['age'].fillna(df['age'].mean(), inplace=True)

# Categorical columns → use mode (most frequent)
df['category'].fillna(df['category'].mode()[0], inplace=True)

# ===== TECHNIQUE 2: Fill with Fixed Value =====

# Known default
df['has_insurance'].fillna(0, inplace=True)
df['country'].fillna('Unknown', inplace=True)

# ===== TECHNIQUE 3: Forward/Backward Fill (Time Series) =====

# Use previous value (forward fill)
df['price'].fillna(method='ffill', inplace=True)

# Use next value (backward fill)
df['price'].fillna(method='bfill', inplace=True)

# ===== TECHNIQUE 4: Fill with Group Statistics =====

# Fill with group median (more accurate)
df['salary'] = df.groupby('department')['salary'].transform(
    lambda x: x.fillna(x.median())
)

# ===== TECHNIQUE 5: Drop Nulls =====

# Drop rows with ANY null
df_clean = df.dropna()

# Drop rows with nulls in SPECIFIC columns
df_clean = df.dropna(subset=['id', 'date', 'amount'])

# Drop columns with too many nulls (>50%)
threshold = len(df) * 0.5
df_clean = df.dropna(axis=1, thresh=threshold)

# ===== TECHNIQUE 6: Flag Nulls (Keep Info) =====

# Create indicator before filling
df['amount_was_null'] = df['amount'].isnull().astype(int)
df['amount'].fillna(df['amount'].median(), inplace=True)
```

### Real Example: Loan Dataset

```python
# Loan dataset cleaning strategy
df_loans = df.copy()

# LoanAmount: 3.3% null → fill with median
df_loans['LoanAmount'] = df_loans['LoanAmount'].fillna(
    df_loans['LoanAmount'].median()
)

# Credit_History: 8% null → fill with mode (most have good history)
df_loans['Credit_History'] = df_loans['Credit_History'].fillna(
    df_loans['Credit_History'].mode()[0]
)

# Gender: 2.4% null → fill with mode
df_loans['Gender'] = df_loans['Gender'].fillna(
    df_loans['Gender'].mode()[0]
)

# Verify cleaning worked
remaining_nulls = df_loans.isnull().sum().sum()
print(f"✅ Remaining nulls: {remaining_nulls}")
```

---

## Step 3: OUTLIERS - Extreme Values

### Detection Methods

```python
# ===== METHOD 1: IQR (Interquartile Range) =====
# Best for: Most numeric variables
# Rule: Values beyond Q1 - 1.5*IQR or Q3 + 1.5*IQR

def detect_outliers_iqr(df, column):
    """Detect outliers using IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower) | (df[column] > upper)]
    
    print(f"Column: {column}")
    print(f"  Normal range: [{lower:.2f}, {upper:.2f}]")
    print(f"  Outliers: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")
    
    return outliers, lower, upper

# Usage
outliers, low, high = detect_outliers_iqr(df, 'ApplicantIncome')

# ===== METHOD 2: Z-Score =====
# Best for: Normally distributed data
# Rule: Values beyond 3 standard deviations

from scipy import stats

def detect_outliers_zscore(df, column, threshold=3):
    """Detect outliers using Z-Score method"""
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    outliers = df[z_scores > threshold]
    
    print(f"Column: {column}")
    print(f"  Outliers (|z| > {threshold}): {len(outliers)}")
    
    return outliers

# Usage
outliers_z = detect_outliers_zscore(df, 'LoanAmount')

# ===== METHOD 3: Percentile Capping =====
# Best for: When you want to keep all rows

def cap_outliers(df, column, lower_pct=0.01, upper_pct=0.99):
    """Cap outliers at percentiles"""
    lower = df[column].quantile(lower_pct)
    upper = df[column].quantile(upper_pct)
    
    df_capped = df.copy()
    df_capped[column] = df_capped[column].clip(lower, upper)
    
    print(f"✅ {column} capped to [{lower:.2f}, {upper:.2f}]")
    return df_capped

# Usage
df_clean = cap_outliers(df, 'ApplicantIncome')
```

### Treatment Options

```python
# OPTION 1: Remove outliers
df_no_outliers = df[(df['income'] >= lower) & (df['income'] <= upper)]

# OPTION 2: Cap outliers (Winsorizing)
df['income'] = df['income'].clip(lower=lower, upper=upper)

# OPTION 3: Flag outliers (keep them, mark them)
df['income_outlier'] = ((df['income'] < lower) | (df['income'] > upper)).astype(int)

# OPTION 4: Log transformation (reduces skewness)
df['income_log'] = np.log1p(df['income'])  # log(x + 1) handles zeros
```

### Visualize Outliers

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Before cleaning
axes[0].boxplot(df['income'].dropna())
axes[0].set_title('Before: Boxplot with Outliers')

# Histogram
df['income'].hist(bins=50, ax=axes[1])
axes[1].set_title('Distribution (skewed = outliers likely)')

# After capping
df_clean['income'].hist(bins=50, ax=axes[2], color='green')
axes[2].set_title('After: Distribution Cleaned')

plt.tight_layout()
plt.show()
```

---

## Step 4: DUPLICATES - Remove Repeated Records

```python
# Check duplicates
print(f"Duplicate rows: {df.duplicated().sum()}")

# View duplicate rows
duplicate_rows = df[df.duplicated(keep=False)]
print(duplicate_rows.head())

# Remove ALL duplicates
df_clean = df.drop_duplicates()

# Remove duplicates based on key columns
df_clean = df.drop_duplicates(subset=['customer_id', 'date'])

# Keep FIRST occurrence
df_clean = df.drop_duplicates(keep='first')

# Keep LAST occurrence
df_clean = df.drop_duplicates(keep='last')

print(f"Before: {len(df)} rows")
print(f"After: {len(df_clean)} rows")
print(f"Removed: {len(df) - len(df_clean)} duplicates")
```

---

## Step 5: FORMAT - Standardize Types and Values

```python
# ===== FIX DATA TYPES =====

# String → Integer
df['age'] = df['age'].astype(int)

# String → Float
df['salary'] = df['salary'].astype(float)

# String → Datetime
df['date'] = pd.to_datetime(df['date'])
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

# Integer → Category
df['risk_level'] = df['risk_level'].astype('category')

# ===== STANDARDIZE TEXT =====

# Lowercase everything
df['name'] = df['name'].str.lower()

# Remove whitespace
df['name'] = df['name'].str.strip()

# Replace values
df['status'] = df['status'].replace({'Y': 'Yes', 'N': 'No'})
df['gender'] = df['gender'].str.replace('M', 'Male')

# ===== STANDARDIZE NUMBERS =====

# Round to 2 decimals
df['amount'] = df['amount'].round(2)

# Convert thousands
df['amount_thousands'] = df['amount'] / 1000

# ===== EXTRACT FROM STRINGS =====

# Extract numbers from strings
df['age_numeric'] = df['age_string'].str.extract('(\d+)').astype(float)

# Extract from dates
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
```

---

## Complete Cleaning Pipeline

```python
def clean_dataset(df, target_col=None):
    """
    Complete data cleaning pipeline
    
    Parameters:
    -----------
    df : pd.DataFrame - Raw dataset
    target_col : str - Target column name (optional)
    
    Returns:
    --------
    pd.DataFrame - Cleaned dataset
    dict - Cleaning summary
    """
    
    print("=" * 60)
    print("STARTING DATA CLEANING PIPELINE")
    print("=" * 60)
    
    df_clean = df.copy()
    original_shape = df.shape
    
    # Step 1: Remove duplicates
    dups_before = df_clean.duplicated().sum()
    df_clean = df_clean.drop_duplicates()
    print(f"\n✅ Duplicates removed: {dups_before}")
    
    # Step 2: Handle nulls
    for col in df_clean.columns:
        null_count = df_clean[col].isnull().sum()
        if null_count > 0:
            null_pct = null_count / len(df_clean) * 100
            
            if null_pct > 50:
                # Too many nulls - consider dropping
                print(f"⚠️  {col}: {null_pct:.1f}% null - review needed")
            else:
                # Fill based on type
                if df_clean[col].dtype in ['int64', 'float64']:
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                    print(f"✅ {col}: filled {null_count} nulls with median")
                else:
                    df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
                    print(f"✅ {col}: filled {null_count} nulls with mode")
    
    # Step 3: Flag outliers in numeric columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    if target_col:
        numeric_cols = [c for c in numeric_cols if c != target_col]
    
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_count = ((df_clean[col] < Q1 - 1.5*IQR) | 
                         (df_clean[col] > Q3 + 1.5*IQR)).sum()
        if outlier_count > 0:
            df_clean[f'{col}_outlier'] = ((df_clean[col] < Q1 - 1.5*IQR) | 
                                           (df_clean[col] > Q3 + 1.5*IQR)).astype(int)
            print(f"🚩 {col}: flagged {outlier_count} outliers")
    
    # Summary
    summary = {
        'original_rows': original_shape[0],
        'cleaned_rows': len(df_clean),
        'removed_rows': original_shape[0] - len(df_clean),
        'original_columns': original_shape[1],
        'final_columns': df_clean.shape[1],
        'remaining_nulls': df_clean.isnull().sum().sum()
    }
    
    print(f"\n{'=' * 60}")
    print("CLEANING SUMMARY")
    print(f"{'=' * 60}")
    print(f"Rows: {summary['original_rows']:,} → {summary['cleaned_rows']:,}")
    print(f"Columns: {summary['original_columns']} → {summary['final_columns']}")
    print(f"Remaining nulls: {summary['remaining_nulls']}")
    print(f"✅ Pipeline completed")
    
    return df_clean, summary


# Usage
df_cleaned, report = clean_dataset(df, target_col='Loan_Status')
```

---

## Cleaning Checklist

Before modeling or analysis, verify:

- [ ] No missing values (or handled intentionally)
- [ ] No duplicate rows
- [ ] Correct data types for all columns
- [ ] Outliers detected and treated
- [ ] Consistent text format (lowercase, no spaces)
- [ ] Dates in datetime format
- [ ] Numeric columns are actually numeric
- [ ] Target column is clean and balanced

---

## Common Mistakes to Avoid

| ❌ Wrong | ✅ Right |
|---------|---------|
| Fill nulls with 0 (always) | Choose fill strategy based on data context |
| Delete all outliers | Investigate first - they might be valid |
| Ignore duplicates | Always check for them |
| Fill then split data | Split first, then fill (prevent data leakage) |

---

## Next Steps

→ [Feature Engineering Guide](06_feature_engineering.md)
