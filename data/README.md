# 📊 Sample Datasets

## Available Datasets

### 1. sample_loan_data.csv
**Description:** Loan approval dataset  
**Rows:** 614  
**Columns:** 13  
**Use case:** Binary classification, credit risk analysis  

**Columns:**
- `Loan_ID` - Unique identifier
- `Gender` - Male/Female
- `Married` - Yes/No
- `Dependents` - Number of dependents
- `Education` - Graduate/Not Graduate
- `Self_Employed` - Yes/No
- `ApplicantIncome` - Income of applicant
- `CoapplicantIncome` - Income of co-applicant
- `LoanAmount` - Loan amount requested
- `Loan_Amount_Term` - Term in months
- `Credit_History` - 1=good, 0=bad
- `Property_Area` - Urban/Semiurban/Rural
- `Loan_Status` - Y=Approved, N=Rejected (TARGET)

**Example usage:**
```python
import pandas as pd
df = pd.read_csv('data/sample_loan_data.csv')
```

---

### 2. sample_transactions.csv
**Description:** Customer transaction data  
**Rows:** 1000  
**Use case:** RFM segmentation, customer analysis  

**Columns:**
- `customer_id` - Unique customer identifier
- `transaction_date` - Date of transaction
- `amount` - Transaction amount
- `category` - Product category
- `payment_method` - Cash/Card/Online

---

## Data Sources

All datasets are either:
1. Synthetic data generated for educational purposes
2. Publicly available datasets with proper attribution
3. Anonymized real-world data

## Usage Guidelines

✅ **Allowed:**
- Educational purposes
- Portfolio projects
- Practice and learning
- Code examples

❌ **Not allowed:**
- Commercial use without proper licensing
- Claiming as original data
- Redistribution without attribution

## Contributing Data

Have a good dataset for examples? Submit a PR with:
- CSV file
- Description in this README
- Data dictionary
- Source/license information
