# 📊 Scikit-Learn Metrics Cheat Sheet

> Quick reference for model evaluation metrics in Python

---

## Import All Metrics

```python
from sklearn.metrics import (
    # Classification
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    
    # Regression
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
```

---

## Classification Metrics

### Confusion Matrix - The Foundation

```
                 PREDICTED
                 NO      YES
ACTUAL  NO    [ TN  |  FP ]
        YES   [ FN  |  TP ]

TN = True Negative  → Correctly predicted NO
TP = True Positive  → Correctly predicted YES
FP = False Positive → Predicted YES, actually NO  (Type I Error)
FN = False Negative → Predicted NO, actually YES  (Type II Error)
```

```python
cm = confusion_matrix(y_true, y_pred)
print(cm)

# Visualize
import seaborn as sns
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

---

### Core Metrics

#### Accuracy
```python
# What % of predictions were correct?
accuracy = accuracy_score(y_true, y_pred)
# Formula: (TP + TN) / (TP + TN + FP + FN)

# When to use: Balanced classes (50/50)
# When NOT to use: Imbalanced classes (95% class A, 5% class B)
# Example: 0.85 = 85% correct predictions
```

#### Precision
```python
# Of all predicted YES, how many were actually YES?
precision = precision_score(y_true, y_pred)
# Formula: TP / (TP + FP)

# When to use: Cost of False Positives is high
# Example (spam): Don't mark real emails as spam
# Example (fraud): Don't block legitimate transactions
# Value: 0.90 = 90% of flagged fraud cases were real fraud
```

#### Recall (Sensitivity)
```python
# Of all actual YES, how many did we find?
recall = recall_score(y_true, y_pred)
# Formula: TP / (TP + FN)

# When to use: Cost of False Negatives is high
# Example (disease): Don't miss sick patients
# Example (fraud): Don't miss fraudulent transactions
# Value: 0.80 = Found 80% of all actual fraud cases
```

#### F1-Score
```python
# Balance between Precision and Recall
f1 = f1_score(y_true, y_pred)
# Formula: 2 * (Precision * Recall) / (Precision + Recall)

# When to use: When Precision and Recall are equally important
# When NOT to use: When one is clearly more important
# Value: 0 = worst, 1 = perfect
```

#### Complete Report
```python
# Get all metrics at once
print(classification_report(y_true, y_pred, target_names=['No', 'Yes']))

# Output:
#               precision  recall  f1-score  support
# No               0.87    0.91      0.89     100
# Yes              0.78    0.71      0.74      50
# accuracy                           0.84     150
```

---

### ROC-AUC Score

```python
# How well does model separate classes?
# Requires probability scores, not just predictions

y_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
auc = roc_auc_score(y_true, y_proba)

# Interpretation:
# 0.5 = Random model (useless)
# 0.7 = Acceptable
# 0.8 = Good
# 0.9 = Excellent
# 1.0 = Perfect (likely overfitting)
```

```python
# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_true, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}', color='blue', lw=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

### Multi-class Classification

```python
# For 3+ classes, use average parameter
precision = precision_score(y_true, y_pred, average='macro')   # Equal weight per class
recall = recall_score(y_true, y_pred, average='weighted')       # Weight by support
f1 = f1_score(y_true, y_pred, average='micro')                  # Global TP/FP/FN
```

---

## Regression Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
```

#### MAE - Mean Absolute Error
```python
mae = mean_absolute_error(y_true, y_pred)
# Average absolute difference
# Same units as target variable
# Less sensitive to outliers than MSE
# Easy to interpret: "On average, off by X units"
```

#### MSE - Mean Squared Error
```python
mse = mean_squared_error(y_true, y_pred)
# Average squared difference
# Penalizes large errors more
# Not in original units (squared)
```

#### RMSE - Root Mean Squared Error
```python
rmse = mean_squared_error(y_true, y_pred, squared=False)
# Square root of MSE
# Same units as target variable
# More interpretable than MSE
# Most common regression metric
```

#### R² - Coefficient of Determination
```python
r2 = r2_score(y_true, y_pred)
# % of variance explained by model
# 1.0 = perfect prediction
# 0.0 = model no better than mean
# <0.0 = model worse than mean
# Example: 0.78 = model explains 78% of variance
```

---

## When to Use Which Metric

| Scenario | Best Metric | Why |
|----------|-------------|-----|
| Balanced classes | Accuracy | Simple and intuitive |
| Imbalanced classes | F1 / AUC-ROC | Accuracy misleads |
| Fraud detection | Recall | Don't miss fraud (FN costly) |
| Spam filter | Precision | Don't block real emails (FP costly) |
| Medical diagnosis | Recall | Don't miss disease (FN = danger) |
| Credit approval | F1 / AUC | Balance both types of error |
| Comparing models | AUC-ROC | Threshold-independent |
| Regression | RMSE + R² | Standard combination |

---

## Imbalanced Classes Problem

```python
# When classes are very unbalanced:
# 95% Class 0, 5% Class 1
# A model predicting always 0 gets 95% accuracy (useless!)

# Solutions:
# 1. Use better metrics (F1, AUC-ROC, not accuracy)
# 2. Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y  # Key: stratify=y
)

# 3. Class weights
model = LogisticRegression(class_weight='balanced')

# 4. SMOTE - Oversample minority class
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
```

---

## Cross-Validation

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

# K-Fold Cross-Validation (5 folds)
cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')

print(f"CV Scores: {cv_scores}")
print(f"Mean: {cv_scores.mean():.3f}")
print(f"Std:  {cv_scores.std():.3f}")
print(f"95% CI: [{cv_scores.mean() - 2*cv_scores.std():.3f}, "
      f"{cv_scores.mean() + 2*cv_scores.std():.3f}]")

# Stratified (preserves class proportions)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_stratified = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
```

---

## Complete Evaluation Template

```python
def evaluate_classifier(model, X_test, y_test, model_name="Model"):
    """
    Complete evaluation of a classifier.
    
    Parameters:
    -----------
    model : sklearn classifier (fitted)
    X_test : features
    y_test : true labels
    model_name : str
    
    Returns:
    --------
    dict : All evaluation metrics
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Print report
    print(f"\n{'=' * 60}")
    print(f"EVALUATION REPORT: {model_name}")
    print(f"{'=' * 60}")
    print(f"\n{classification_report(y_test, y_pred)}")
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
    }
    
    if y_proba is not None:
        metrics['auc_roc'] = roc_auc_score(y_test, y_proba)
    
    print("Summary:")
    for metric, value in metrics.items():
        print(f"  {metric.upper():12}: {value:.4f}")
    
    # Confusion Matrix plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title(f'Confusion Matrix - {model_name}')
    axes[0].set_ylabel('Actual')
    axes[0].set_xlabel('Predicted')
    
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        axes[1].plot(fpr, tpr, color='blue', lw=2,
                    label=f"AUC = {metrics['auc_roc']:.3f}")
        axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return metrics


# Usage
results = evaluate_classifier(model, X_test, y_test, "Logistic Regression")
```

---

## Quick Reference Card

```python
# ONE LINE EVALUATION
print(classification_report(y_true, y_pred))           # Full report
print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")  # Accuracy only
print(f"AUC: {roc_auc_score(y_true, y_proba):.3f}")       # AUC-ROC

# CONFUSION MATRIX
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

# METRICS FROM CONFUSION MATRIX
accuracy  = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall    = tp / (tp + fn)
f1        = 2 * precision * recall / (precision + recall)
```

---

## Next Steps

→ [ML Basics Guide](08_ml_basics.md)
