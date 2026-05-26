"""
data_quality_check.py
======================
Author: Fredys Caballero
Version: 1.0

Automated data quality checks with pass/fail scoring.
Use before loading data into dashboards or models.

Usage:
    from scripts.data_quality_check import DataQualityChecker
    
    checker = DataQualityChecker(df)
    report = checker.run_all_checks()
    checker.print_report()
"""

import pandas as pd
import numpy as np
from datetime import datetime


class DataQualityChecker:
    """
    Automated data quality checker.
    
    Runs checks on:
    - Completeness  (missing values)
    - Uniqueness    (duplicates)
    - Validity      (data types, ranges)
    - Consistency   (format standardization)
    - Timeliness    (date freshness, if applicable)
    
    Each check returns PASS, WARNING, or FAIL.
    
    Usage:
    ------
    checker = DataQualityChecker(df)
    report = checker.run_all_checks()
    checker.print_report()
    score = checker.get_score()
    """
    
    def __init__(self, df, dataset_name="Dataset"):
        self.df = df.copy()
        self.name = dataset_name
        self.results = []
        self.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # =========================================================================
    # COMPLETENESS CHECKS
    # =========================================================================
    
    def check_no_nulls(self, columns=None, threshold_warn=0.05, threshold_fail=0.20):
        """
        Check for missing values.
        
        PASS    : < threshold_warn% missing
        WARNING : threshold_warn% - threshold_fail% missing
        FAIL    : > threshold_fail% missing
        """
        
        cols = columns or self.df.columns
        
        for col in cols:
            null_pct = self.df[col].isnull().sum() / len(self.df)
            null_count = self.df[col].isnull().sum()
            
            if null_pct == 0:
                status = 'PASS'
                message = f"No missing values"
            elif null_pct < threshold_warn:
                status = 'PASS'
                message = f"{null_count} missing ({null_pct*100:.1f}%) - acceptable"
            elif null_pct < threshold_fail:
                status = 'WARNING'
                message = f"{null_count} missing ({null_pct*100:.1f}%) - review needed"
            else:
                status = 'FAIL'
                message = f"{null_count} missing ({null_pct*100:.1f}%) - action required"
            
            self.results.append({
                'Check': 'Completeness',
                'Column': col,
                'Status': status,
                'Detail': message
            })
        
        return self
    
    # =========================================================================
    # UNIQUENESS CHECKS
    # =========================================================================
    
    def check_no_duplicates(self, subset=None):
        """Check for duplicate rows."""
        
        dups = self.df.duplicated(subset=subset).sum()
        dup_pct = dups / len(self.df) * 100
        
        if dups == 0:
            status = 'PASS'
            message = "No duplicate rows"
        elif dups <= 5:
            status = 'WARNING'
            message = f"{dups} duplicates found ({dup_pct:.2f}%) - check if expected"
        else:
            status = 'FAIL'
            message = f"{dups} duplicates found ({dup_pct:.2f}%) - action required"
        
        self.results.append({
            'Check': 'Uniqueness',
            'Column': 'ALL' if subset is None else str(subset),
            'Status': status,
            'Detail': message
        })
        
        return self
    
    def check_unique_column(self, column, should_be_unique=True):
        """Check if a column should have all unique values (e.g., ID columns)."""
        
        unique_count = self.df[column].nunique()
        total_count = len(self.df)
        duplicates = total_count - unique_count
        
        if should_be_unique:
            if duplicates == 0:
                status = 'PASS'
                message = f"All {unique_count:,} values are unique"
            else:
                status = 'FAIL'
                message = f"{duplicates} duplicate values found - ID column should be unique"
        else:
            status = 'PASS'
            message = f"{unique_count} unique values out of {total_count} rows"
        
        self.results.append({
            'Check': 'Uniqueness',
            'Column': column,
            'Status': status,
            'Detail': message
        })
        
        return self
    
    # =========================================================================
    # VALIDITY CHECKS
    # =========================================================================
    
    def check_value_range(self, column, min_val=None, max_val=None):
        """Check if numeric values are within expected range."""
        
        if column not in self.df.columns:
            self.results.append({
                'Check': 'Validity',
                'Column': column,
                'Status': 'WARNING',
                'Detail': f"Column not found in dataset"
            })
            return self
        
        actual_min = self.df[column].min()
        actual_max = self.df[column].max()
        
        violations = 0
        
        if min_val is not None:
            below_min = (self.df[column] < min_val).sum()
            violations += below_min
        
        if max_val is not None:
            above_max = (self.df[column] > max_val).sum()
            violations += above_max
        
        if violations == 0:
            status = 'PASS'
            message = f"Range OK: [{actual_min:.2f}, {actual_max:.2f}]"
        else:
            status = 'FAIL'
            message = f"{violations} values outside expected range [{min_val}, {max_val}]"
        
        self.results.append({
            'Check': 'Validity',
            'Column': column,
            'Status': status,
            'Detail': message
        })
        
        return self
    
    def check_allowed_values(self, column, allowed_values):
        """Check if categorical column only contains allowed values."""
        
        if column not in self.df.columns:
            return self
        
        actual_values = set(self.df[column].dropna().unique())
        allowed_set = set(allowed_values)
        unexpected = actual_values - allowed_set
        
        if not unexpected:
            status = 'PASS'
            message = f"All values are in allowed list"
        elif len(unexpected) <= 3:
            status = 'WARNING'
            message = f"Unexpected values found: {unexpected}"
        else:
            status = 'FAIL'
            message = f"{len(unexpected)} unexpected values: {list(unexpected)[:5]}..."
        
        self.results.append({
            'Check': 'Validity',
            'Column': column,
            'Status': status,
            'Detail': message
        })
        
        return self
    
    def check_data_types(self, type_expectations):
        """
        Check if columns have expected data types.
        
        Parameters:
        -----------
        type_expectations : dict
            {'column': 'expected_type'} 
            Types: 'numeric', 'categorical', 'datetime', 'boolean'
        """
        
        type_map = {
            'numeric': ['int64', 'float64', 'int32', 'float32'],
            'categorical': ['object', 'category'],
            'datetime': ['datetime64[ns]', 'datetime64'],
            'boolean': ['bool']
        }
        
        for col, expected_type in type_expectations.items():
            if col not in self.df.columns:
                continue
            
            actual_type = str(self.df[col].dtype)
            allowed_types = type_map.get(expected_type, [expected_type])
            
            is_correct = any(allowed in actual_type for allowed in allowed_types)
            
            if is_correct:
                status = 'PASS'
                message = f"Type OK: {actual_type}"
            else:
                status = 'FAIL'
                message = f"Expected {expected_type}, got {actual_type}"
            
            self.results.append({
                'Check': 'Data Types',
                'Column': col,
                'Status': status,
                'Detail': message
            })
        
        return self
    
    def check_no_negative(self, columns):
        """Check that specific columns have no negative values."""
        
        for col in columns:
            if col not in self.df.columns:
                continue
            
            neg_count = (self.df[col] < 0).sum()
            
            if neg_count == 0:
                status = 'PASS'
                message = "No negative values"
            else:
                status = 'FAIL'
                message = f"{neg_count} negative values found"
            
            self.results.append({
                'Check': 'Validity',
                'Column': col,
                'Status': status,
                'Detail': message
            })
        
        return self
    
    # =========================================================================
    # CONSISTENCY CHECKS
    # =========================================================================
    
    def check_no_outliers(self, columns, threshold=3):
        """Check for extreme outliers using Z-Score."""
        
        from scipy import stats
        
        for col in columns:
            if col not in self.df.columns:
                continue
            
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                continue
            
            clean_data = self.df[col].dropna()
            z_scores = np.abs(stats.zscore(clean_data))
            extreme_count = (z_scores > threshold).sum()
            extreme_pct = extreme_count / len(clean_data) * 100
            
            if extreme_count == 0:
                status = 'PASS'
                message = f"No extreme outliers (|z| > {threshold})"
            elif extreme_pct < 1:
                status = 'WARNING'
                message = f"{extreme_count} extreme values ({extreme_pct:.2f}%) - investigate"
            else:
                status = 'FAIL'
                message = f"{extreme_count} extreme values ({extreme_pct:.2f}%) - action needed"
            
            self.results.append({
                'Check': 'Consistency',
                'Column': col,
                'Status': status,
                'Detail': message
            })
        
        return self
    
    def check_date_freshness(self, date_column, max_days_old=30):
        """Check if date column has recent data."""
        
        if date_column not in self.df.columns:
            return self
        
        try:
            dates = pd.to_datetime(self.df[date_column]).dropna()
            
            if len(dates) == 0:
                return self
            
            max_date = dates.max()
            days_old = (datetime.now() - max_date.to_pydatetime()).days
            
            if days_old <= max_days_old:
                status = 'PASS'
                message = f"Latest date: {max_date.date()} ({days_old} days ago)"
            elif days_old <= max_days_old * 2:
                status = 'WARNING'
                message = f"Data may be stale: {days_old} days since latest record"
            else:
                status = 'FAIL'
                message = f"Data is old: {days_old} days since latest record"
            
            self.results.append({
                'Check': 'Timeliness',
                'Column': date_column,
                'Status': status,
                'Detail': message
            })
        
        except Exception as e:
            self.results.append({
                'Check': 'Timeliness',
                'Column': date_column,
                'Status': 'WARNING',
                'Detail': f"Could not parse dates: {e}"
            })
        
        return self
    
    # =========================================================================
    # RUN ALL BASIC CHECKS
    # =========================================================================
    
    def run_basic_checks(self):
        """
        Run standard checks suitable for any dataset.
        No configuration needed.
        """
        
        # Completeness - all columns
        self.check_no_nulls()
        
        # Uniqueness - full dataset
        self.check_no_duplicates()
        
        # Outliers - all numeric
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            self.check_no_outliers(numeric_cols)
        
        return self
    
    def run_all_checks(self):
        """Alias for run_basic_checks."""
        return self.run_basic_checks()
    
    # =========================================================================
    # REPORTING
    # =========================================================================
    
    def get_results_df(self):
        """Return results as a DataFrame."""
        return pd.DataFrame(self.results)
    
    def get_score(self):
        """
        Calculate overall data quality score (0-100).
        
        PASS    = 100 points
        WARNING = 50 points  
        FAIL    = 0 points
        """
        
        if not self.results:
            return 0
        
        score_map = {'PASS': 100, 'WARNING': 50, 'FAIL': 0}
        scores = [score_map.get(r['Status'], 0) for r in self.results]
        
        return round(sum(scores) / len(scores), 1)
    
    def print_report(self):
        """Print formatted quality report."""
        
        results_df = self.get_results_df()
        score = self.get_score()
        
        print("=" * 70)
        print(f"DATA QUALITY REPORT: {self.name}")
        print(f"Generated: {self.timestamp}")
        print("=" * 70)
        
        # Score summary
        pass_count = (results_df['Status'] == 'PASS').sum()
        warn_count = (results_df['Status'] == 'WARNING').sum()
        fail_count = (results_df['Status'] == 'FAIL').sum()
        total = len(results_df)
        
        print(f"\n📊 OVERALL SCORE: {score}/100")
        print(f"   ✅ PASS:    {pass_count}/{total}")
        print(f"   ⚠️  WARNING: {warn_count}/{total}")
        print(f"   ❌ FAIL:    {fail_count}/{total}")
        
        # Grade
        if score >= 90:
            grade = "🟢 EXCELLENT - Ready for production"
        elif score >= 75:
            grade = "🟡 GOOD - Minor issues to address"
        elif score >= 60:
            grade = "🟠 FAIR - Several issues need attention"
        else:
            grade = "🔴 POOR - Major data quality issues"
        
        print(f"\n   Grade: {grade}")
        
        # Failures first
        if fail_count > 0:
            print(f"\n{'─' * 70}")
            print("❌ FAILURES (Action Required)")
            print(f"{'─' * 70}")
            failures = results_df[results_df['Status'] == 'FAIL']
            for _, row in failures.iterrows():
                print(f"  [{row['Check']}] {row['Column']}: {row['Detail']}")
        
        # Warnings
        if warn_count > 0:
            print(f"\n{'─' * 70}")
            print("⚠️  WARNINGS (Review Recommended)")
            print(f"{'─' * 70}")
            warnings = results_df[results_df['Status'] == 'WARNING']
            for _, row in warnings.iterrows():
                print(f"  [{row['Check']}] {row['Column']}: {row['Detail']}")
        
        # Passes
        if pass_count > 0:
            print(f"\n{'─' * 70}")
            print("✅ PASSED CHECKS")
            print(f"{'─' * 70}")
            passes = results_df[results_df['Status'] == 'PASS']
            for _, row in passes.iterrows():
                print(f"  [{row['Check']}] {row['Column']}: {row['Detail']}")
        
        print(f"\n{'=' * 70}")
        
        return self
    
    def export_report(self, filepath):
        """Save report to CSV."""
        df = self.get_results_df()
        df['Score'] = self.get_score()
        df['Dataset'] = self.name
        df['Timestamp'] = self.timestamp
        df.to_csv(filepath, index=False)
        print(f"💾 Report saved to: {filepath}")
        return self


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    
    # Create sample dataset with quality issues
    np.random.seed(42)
    n = 500
    
    sample_df = pd.DataFrame({
        'loan_id': range(1, n + 1),
        'applicant_income': np.random.gamma(4, 10000, n),
        'loan_amount': np.random.gamma(3, 5000, n),
        'credit_history': np.random.choice([0, 1, np.nan], n, p=[0.1, 0.8, 0.1]),
        'property_area': np.random.choice(['Urban', 'Rural', 'Semiurban', 'Unknown'], n),
        'loan_status': np.random.choice(['Y', 'N'], n)
    })
    
    # Introduce issues
    sample_df.loc[np.random.choice(sample_df.index, 30), 'applicant_income'] = np.nan
    sample_df.loc[np.random.choice(sample_df.index, 3), 'loan_amount'] = -500  # Negatives
    sample_df.loc[np.random.choice(sample_df.index, 5), 'applicant_income'] = 999999  # Outliers
    sample_df = pd.concat([sample_df, sample_df.head(10)])  # Duplicates
    
    # Run checks
    checker = (DataQualityChecker(sample_df, "Loan Applications")
               .check_no_nulls()
               .check_no_duplicates()
               .check_unique_column('loan_id', should_be_unique=True)
               .check_no_negative(['applicant_income', 'loan_amount'])
               .check_value_range('credit_history', min_val=0, max_val=1)
               .check_allowed_values('property_area', ['Urban', 'Rural', 'Semiurban'])
               .check_allowed_values('loan_status', ['Y', 'N'])
               .check_no_outliers(['applicant_income', 'loan_amount']))
    
    checker.print_report()
    
    print(f"\nFinal Score: {checker.get_score()}/100")
