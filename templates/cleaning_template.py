"""
Data Cleaning Template
======================
Author: Fredys Caballero
Version: 1.0

Reusable cleaning functions for any dataset.

Usage:
    from templates.cleaning_template import clean_dataset, handle_nulls, remove_outliers
    df_clean = clean_dataset(df)
"""

import pandas as pd
import numpy as np
from scipy import stats


# =============================================================================
# ASSESSMENT
# =============================================================================

def assess_quality(df):
    """
    Quick data quality report.
    
    Parameters:
    -----------
    df : pd.DataFrame
    
    Returns:
    --------
    dict : Quality metrics
    """
    
    print("=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)
    
    # Shape
    print(f"\n📊 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Missing values
    nulls = df.isnull().sum()
    null_pct = (nulls / len(df) * 100).round(2)
    
    null_df = pd.DataFrame({
        'Missing_Count': nulls,
        'Missing_Pct': null_pct
    }).query('Missing_Count > 0').sort_values('Missing_Count', ascending=False)
    
    if len(null_df) > 0:
        print(f"\n⚠️  Missing values ({len(null_df)} columns affected):")
        print(null_df)
    else:
        print("\n✅ No missing values")
    
    # Duplicates
    dups = df.duplicated().sum()
    print(f"\n🔄 Duplicates: {dups} rows ({dups/len(df)*100:.2f}%)")
    
    # Data types breakdown
    print(f"\n📋 Data types:")
    print(df.dtypes.value_counts())
    
    # Memory
    mem_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"\n💾 Memory usage: {mem_mb:.2f} MB")
    
    return {
        'rows': df.shape[0],
        'columns': df.shape[1],
        'missing_columns': len(null_df),
        'missing_total': nulls.sum(),
        'duplicates': dups,
        'memory_mb': round(mem_mb, 2)
    }


# =============================================================================
# HANDLE NULLS
# =============================================================================

def handle_nulls(df, strategy='auto', custom_fills=None, drop_threshold=0.5):
    """
    Handle missing values in a DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
    strategy : str
        'auto'   - numeric=median, categorical=mode
        'mean'   - fill numeric with mean
        'median' - fill numeric with median
        'drop'   - drop rows with nulls
    custom_fills : dict
        Custom fill values per column: {'col': value}
    drop_threshold : float
        Drop columns with more than this % missing (default 0.5 = 50%)
    
    Returns:
    --------
    pd.DataFrame : Cleaned dataframe
    """
    
    df_clean = df.copy()
    
    # Drop columns with too many nulls
    for col in df_clean.columns:
        null_pct = df_clean[col].isnull().sum() / len(df_clean)
        if null_pct > drop_threshold:
            df_clean.drop(columns=[col], inplace=True)
            print(f"⚠️  Dropped column '{col}': {null_pct*100:.1f}% missing")
    
    # Apply custom fills first
    if custom_fills:
        for col, value in custom_fills.items():
            if col in df_clean.columns:
                filled = df_clean[col].isnull().sum()
                df_clean[col].fillna(value, inplace=True)
                print(f"✅ {col}: filled {filled} nulls with custom value '{value}'")
    
    # Auto-fill remaining nulls
    for col in df_clean.columns:
        null_count = df_clean[col].isnull().sum()
        if null_count == 0:
            continue
        
        if strategy == 'drop':
            df_clean = df_clean.dropna(subset=[col])
            
        elif df_clean[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            if strategy in ['auto', 'median']:
                fill_val = df_clean[col].median()
                df_clean[col].fillna(fill_val, inplace=True)
                print(f"✅ {col}: filled {null_count} nulls with median ({fill_val:.2f})")
            elif strategy == 'mean':
                fill_val = df_clean[col].mean()
                df_clean[col].fillna(fill_val, inplace=True)
                print(f"✅ {col}: filled {null_count} nulls with mean ({fill_val:.2f})")
        else:
            fill_val = df_clean[col].mode()[0]
            df_clean[col].fillna(fill_val, inplace=True)
            print(f"✅ {col}: filled {null_count} nulls with mode ('{fill_val}')")
    
    remaining = df_clean.isnull().sum().sum()
    print(f"\n📊 Remaining nulls: {remaining}")
    
    return df_clean


# =============================================================================
# HANDLE OUTLIERS
# =============================================================================

def detect_outliers_iqr(df, column):
    """
    Detect outliers using IQR method.
    
    Returns:
    --------
    tuple : (outlier_df, lower_bound, upper_bound)
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower) | (df[column] > upper)]
    
    return outliers, lower, upper


def detect_outliers_zscore(df, column, threshold=3):
    """
    Detect outliers using Z-Score method.
    
    Returns:
    --------
    pd.DataFrame : Outlier rows
    """
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    return df[z_scores > threshold]


def remove_outliers(df, columns=None, method='iqr', treatment='flag'):
    """
    Handle outliers in numeric columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
    columns : list - Columns to check (None = all numeric)
    method : str - 'iqr' or 'zscore'
    treatment : str
        'flag'   - Add binary flag column (keeps data)
        'cap'    - Cap at bounds (keeps data, changes values)
        'remove' - Remove outlier rows (loses data)
    
    Returns:
    --------
    pd.DataFrame : Treated dataframe
    """
    
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        if method == 'iqr':
            outliers, lower, upper = detect_outliers_iqr(df_clean, col)
            mask = (df_clean[col] < lower) | (df_clean[col] > upper)
        else:  # zscore
            z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
            mask = z_scores > 3
            lower = df_clean[col].mean() - 3 * df_clean[col].std()
            upper = df_clean[col].mean() + 3 * df_clean[col].std()
        
        outlier_count = mask.sum()
        
        if outlier_count == 0:
            continue
        
        if treatment == 'flag':
            df_clean[f'{col}_outlier'] = mask.astype(int)
            print(f"🚩 {col}: flagged {outlier_count} outliers")
            
        elif treatment == 'cap':
            df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
            print(f"✂️  {col}: capped {outlier_count} values to [{lower:.2f}, {upper:.2f}]")
            
        elif treatment == 'remove':
            df_clean = df_clean[~mask]
            print(f"🗑️  {col}: removed {outlier_count} outlier rows")
    
    return df_clean


# =============================================================================
# DUPLICATES
# =============================================================================

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows.
    
    Parameters:
    -----------
    df : pd.DataFrame
    subset : list - Columns to consider for duplicate detection
    keep : str - 'first', 'last', or False (remove all)
    
    Returns:
    --------
    pd.DataFrame : Deduplicated dataframe
    """
    
    before = len(df)
    df_clean = df.drop_duplicates(subset=subset, keep=keep)
    removed = before - len(df_clean)
    
    print(f"🔄 Duplicates removed: {removed} rows ({removed/before*100:.2f}%)")
    print(f"   Rows: {before:,} → {len(df_clean):,}")
    
    return df_clean


# =============================================================================
# TYPE CONVERSION
# =============================================================================

def fix_data_types(df, type_map=None):
    """
    Fix data types for columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
    type_map : dict - {'column': 'type'} e.g. {'age': 'int', 'date': 'datetime'}
    
    Returns:
    --------
    pd.DataFrame : Fixed dataframe
    """
    
    df_clean = df.copy()
    
    if type_map:
        for col, dtype in type_map.items():
            if col not in df_clean.columns:
                continue
            try:
                if dtype == 'datetime':
                    df_clean[col] = pd.to_datetime(df_clean[col])
                elif dtype == 'int':
                    df_clean[col] = df_clean[col].astype(int)
                elif dtype == 'float':
                    df_clean[col] = df_clean[col].astype(float)
                elif dtype == 'str':
                    df_clean[col] = df_clean[col].astype(str)
                elif dtype == 'category':
                    df_clean[col] = df_clean[col].astype('category')
                
                print(f"✅ {col}: converted to {dtype}")
            except Exception as e:
                print(f"❌ {col}: failed to convert - {e}")
    
    return df_clean


# =============================================================================
# TEXT STANDARDIZATION
# =============================================================================

def standardize_text(df, columns=None):
    """
    Standardize text columns: lowercase, strip whitespace.
    
    Parameters:
    -----------
    df : pd.DataFrame
    columns : list - Text columns to standardize (None = all object columns)
    
    Returns:
    --------
    pd.DataFrame : Standardized dataframe
    """
    
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.select_dtypes(include=['object']).columns.tolist()
    
    for col in columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].str.strip().str.lower()
            print(f"✅ {col}: standardized (lowercase + strip)")
    
    return df_clean


# =============================================================================
# COMPLETE PIPELINE
# =============================================================================

def clean_dataset(df, 
                  target_col=None,
                  null_strategy='auto',
                  custom_fills=None,
                  outlier_treatment='flag',
                  remove_dups=True,
                  drop_null_threshold=0.5):
    """
    Complete data cleaning pipeline.
    
    Parameters:
    -----------
    df : pd.DataFrame - Raw dataset
    target_col : str - Target column (excluded from outlier treatment)
    null_strategy : str - 'auto', 'mean', 'median', 'drop'
    custom_fills : dict - Custom null fills per column
    outlier_treatment : str - 'flag', 'cap', 'remove'
    remove_dups : bool - Whether to remove duplicates
    drop_null_threshold : float - Drop columns with > this % missing
    
    Returns:
    --------
    tuple : (pd.DataFrame, dict) - Cleaned df and summary report
    """
    
    print("=" * 60)
    print("STARTING CLEANING PIPELINE")
    print("=" * 60)
    
    df_clean = df.copy()
    original_shape = df.shape
    
    # Step 1: Remove duplicates
    if remove_dups:
        print("\n📌 STEP 1: Removing duplicates")
        df_clean = remove_duplicates(df_clean)
    
    # Step 2: Handle nulls
    print("\n📌 STEP 2: Handling missing values")
    df_clean = handle_nulls(
        df_clean,
        strategy=null_strategy,
        custom_fills=custom_fills,
        drop_threshold=drop_null_threshold
    )
    
    # Step 3: Handle outliers
    print("\n📌 STEP 3: Treating outliers")
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    df_clean = remove_outliers(df_clean, columns=numeric_cols, treatment=outlier_treatment)
    
    # Summary
    summary = {
        'original_rows': original_shape[0],
        'final_rows': len(df_clean),
        'rows_removed': original_shape[0] - len(df_clean),
        'original_columns': original_shape[1],
        'final_columns': df_clean.shape[1],
        'remaining_nulls': df_clean.isnull().sum().sum()
    }
    
    print(f"\n{'=' * 60}")
    print("CLEANING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Rows:    {summary['original_rows']:,} → {summary['final_rows']:,}")
    print(f"Columns: {summary['original_columns']} → {summary['final_columns']}")
    print(f"Nulls remaining: {summary['remaining_nulls']}")
    print(f"✅ Dataset ready for analysis")
    
    return df_clean, summary


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    
    # Create sample dirty dataset
    np.random.seed(42)
    n = 500
    
    dirty_df = pd.DataFrame({
        'id': range(1, n+1),
        'age': np.random.randint(18, 70, n),
        'income': np.random.gamma(4, 10000, n),
        'loan_amount': np.random.gamma(3, 5000, n),
        'category': np.random.choice(['A', 'B', 'C', None], n),
        'status': np.random.choice(['active', 'inactive', 'ACTIVE'], n)
    })
    
    # Introduce issues
    dirty_df.loc[np.random.choice(dirty_df.index, 50), 'income'] = np.nan
    dirty_df.loc[np.random.choice(dirty_df.index, 20), 'loan_amount'] = np.nan
    dirty_df.loc[np.random.choice(dirty_df.index, 5), 'income'] = 999999  # Outliers
    dirty_df = pd.concat([dirty_df, dirty_df.head(10)])  # Duplicates
    
    print("=== BEFORE CLEANING ===")
    assess_quality(dirty_df)
    
    print("\n\n=== CLEANING PIPELINE ===")
    clean_df, report = clean_dataset(
        dirty_df,
        custom_fills={'category': 'Unknown'},
        outlier_treatment='cap'
    )
    
    print("\n\n=== AFTER CLEANING ===")
    assess_quality(clean_df)
    
    print("\n\nReport:", report)
