# 🔍 Exploratory Data Analysis (EDA) Guide

## What is EDA?

Exploratory Data Analysis is the critical first step in understanding your data before any analysis or modeling.

## The 5 Essential Steps

### 1. Preview Data
```python
df.head()        # First 5 rows
df.tail()        # Last 5 rows
df.sample(10)    # Random 10 rows
```

**Why:** See what you're working with

---

### 2. Check Structure
```python
df.info()        # Types, memory, nulls
df.shape         # (rows, columns)
df.columns       # Column names
```

**Why:** Understand data types and size

---

### 3. Summary Statistics
```python
df.describe()    # Mean, median, std, min, max
```

**Why:** Detect outliers, understand distributions

---

### 4. Find Missing Values
```python
df.isnull().sum()
(df.isnull().sum() / len(df) * 100).round(2)  # Percentage
```

**Why:** Plan cleaning strategy

---

### 5. Check Distributions
```python
df['column'].value_counts()              # For categorical
df['column'].hist(bins=30)               # For numeric
```

**Why:** Understand data balance and patterns

---

## Complete EDA Template

```python
def quick_eda(df):
    """Run complete EDA in 10 seconds"""
    
    print(f"Shape: {df.shape}")
    print(f"\nNulls:\n{df.isnull().sum()}")
    print(f"\nStats:\n{df.describe()}")
    
    # Visualize first numeric column
    numeric_col = df.select_dtypes(include=['number']).columns[0]
    df[numeric_col].hist(bins=30)
    plt.title(f'Distribution of {numeric_col}')
    plt.show()

# Usage
quick_eda(df)
```

---

## Visualization Checklist

- [ ] Histogram for each numeric variable
- [ ] Bar chart for categorical variables
- [ ] Correlation heatmap
- [ ] Boxplots to detect outliers
- [ ] Scatter plots for relationships

---

## Common Patterns to Look For

### 1. Skewed Distributions
- **Problem:** Most values on one side
- **Action:** Consider log transformation

### 2. Bimodal Distributions
- **Problem:** Two peaks
- **Action:** May indicate distinct groups

### 3. High Correlation
- **Problem:** Two variables move together
- **Action:** May cause multicollinearity in models

### 4. Class Imbalance
- **Problem:** 90% class A, 10% class B
- **Action:** Use stratified sampling or SMOTE

---

## Next Steps

→ [Data Cleaning Guide](05_data_cleaning.md)
