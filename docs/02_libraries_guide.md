# 📚 Python Libraries Guide

## Essential Libraries for Data Analysis

### Core Libraries

#### 1. pandas - Data Manipulation
```python
import pandas as pd

# What it does: Excel on steroids
# Use for: Loading, cleaning, transforming data
```

**Key Functions:**
- `pd.read_csv()` - Load data
- `df.head()` - Preview data
- `df.groupby()` - Aggregate
- `df.merge()` - Join tables

**Example:**
```python
df = pd.read_csv('data.csv')
summary = df.groupby('category')['sales'].sum()
```

---

#### 2. numpy - Mathematical Operations
```python
import numpy as np

# What it does: Fast math operations
# Use for: Calculations, arrays, statistics
```

**Key Functions:**
- `np.mean()` - Average
- `np.std()` - Standard deviation
- `np.random.randint()` - Random numbers

**Example:**
```python
outliers = data[data > np.mean(data) + 3*np.std(data)]
```

---

#### 3. matplotlib - Basic Plotting
```python
import matplotlib.pyplot as plt

# What it does: Create charts
# Use for: Quick visualizations
```

**Example:**
```python
plt.hist(df['column'], bins=30)
plt.title('Distribution')
plt.show()
```

---

#### 4. seaborn - Beautiful Plots
```python
import seaborn as sns

# What it does: Professional charts
# Use for: Presentations, reports
```

**Example:**
```python
sns.heatmap(correlation_matrix, annot=True)
plt.show()
```

---

### Machine Learning Libraries

#### 5. scikit-learn
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# What it does: Machine learning made easy
# Use for: Predictions, classifications
```

---

## When to Use What

| Task | Library | Function |
|------|---------|----------|
| Load CSV | pandas | `pd.read_csv()` |
| Calculate mean | numpy | `np.mean()` |
| Create chart | matplotlib | `plt.plot()` |
| Heatmap | seaborn | `sns.heatmap()` |
| Train model | sklearn | `LogisticRegression()` |

---

## Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Next Steps

→ [Data Loading Guide](03_data_loading.md)
