"""
quick_eda.py
============
Author: Fredys Caballero
Version: 1.0

Run a complete EDA on any CSV file from the command line.

Usage:
    python quick_eda.py data.csv
    python quick_eda.py data.csv --target Loan_Status
    python quick_eda.py data.csv --target Loan_Status --output report.txt
    
Features:
    - Auto-detects column types
    - Handles missing values
    - Generates visualizations
    - Exports text report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
import os
from datetime import datetime


def load_data(filepath):
    """Load data from CSV file with error handling."""
    
    if not os.path.exists(filepath):
        print(f"❌ Error: File '{filepath}' not found")
        sys.exit(1)
    
    ext = os.path.splitext(filepath)[1].lower()
    
    try:
        if ext == '.csv':
            df = pd.read_csv(filepath)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath)
        elif ext == '.json':
            df = pd.read_json(filepath)
        else:
            # Try CSV as default
            df = pd.read_csv(filepath)
        
        print(f"✅ Loaded: {filepath}")
        print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        return df
    
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        sys.exit(1)


def run_quick_eda(df, target_col=None, output_file=None):
    """
    Run complete EDA and optionally save report.
    
    Parameters:
    -----------
    df : pd.DataFrame
    target_col : str, optional - Name of target column
    output_file : str, optional - Path to save text report
    """
    
    lines = []
    
    def log(text=""):
        print(text)
        lines.append(text)
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    log("=" * 70)
    log("QUICK EDA REPORT")
    log(f"Generated: {timestamp}")
    log("=" * 70)
    
    # =========================================================================
    # SECTION 1: OVERVIEW
    # =========================================================================
    log("\n" + "─" * 70)
    log("1. OVERVIEW")
    log("─" * 70)
    
    log(f"Rows:    {df.shape[0]:,}")
    log(f"Columns: {df.shape[1]}")
    
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    log(f"Memory:  {memory_mb:.2f} MB")
    
    # Column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    log(f"\nColumn Types:")
    log(f"  Numeric:     {len(numeric_cols)} → {numeric_cols[:5]}{'...' if len(numeric_cols) > 5 else ''}")
    log(f"  Categorical: {len(categorical_cols)} → {categorical_cols[:5]}{'...' if len(categorical_cols) > 5 else ''}")
    log(f"  Datetime:    {len(datetime_cols)} → {datetime_cols}")
    
    # =========================================================================
    # SECTION 2: MISSING VALUES
    # =========================================================================
    log("\n" + "─" * 70)
    log("2. MISSING VALUES")
    log("─" * 70)
    
    nulls = df.isnull().sum()
    null_pct = (nulls / len(df) * 100).round(2)
    null_df = pd.DataFrame({
        'Count': nulls,
        'Percentage': null_pct
    }).query('Count > 0').sort_values('Count', ascending=False)
    
    if len(null_df) > 0:
        log(f"Columns with missing values: {len(null_df)}")
        log(f"Total missing cells: {nulls.sum():,}")
        log("")
        log(null_df.to_string())
    else:
        log("✅ No missing values found")
    
    # =========================================================================
    # SECTION 3: DUPLICATES
    # =========================================================================
    log("\n" + "─" * 70)
    log("3. DUPLICATES")
    log("─" * 70)
    
    dups = df.duplicated().sum()
    log(f"Duplicate rows: {dups} ({dups/len(df)*100:.2f}%)")
    
    # =========================================================================
    # SECTION 4: SUMMARY STATISTICS
    # =========================================================================
    log("\n" + "─" * 70)
    log("4. SUMMARY STATISTICS (Numeric)")
    log("─" * 70)
    
    if len(numeric_cols) > 0:
        stats = df[numeric_cols].describe().round(3)
        log(stats.to_string())
    else:
        log("No numeric columns found")
    
    # =========================================================================
    # SECTION 5: CATEGORICAL DISTRIBUTIONS
    # =========================================================================
    log("\n" + "─" * 70)
    log("5. CATEGORICAL DISTRIBUTIONS (Top values)")
    log("─" * 70)
    
    for col in categorical_cols[:8]:  # Limit to 8 columns
        log(f"\n{col} ({df[col].nunique()} unique values):")
        top_values = df[col].value_counts().head(5)
        for val, count in top_values.items():
            pct = count / len(df) * 100
            log(f"  {str(val):<20} {count:>6} ({pct:.1f}%)")
    
    # =========================================================================
    # SECTION 6: TARGET DISTRIBUTION (if provided)
    # =========================================================================
    if target_col and target_col in df.columns:
        log("\n" + "─" * 70)
        log(f"6. TARGET COLUMN: {target_col}")
        log("─" * 70)
        
        target_dist = df[target_col].value_counts()
        target_pct = (df[target_col].value_counts(normalize=True) * 100).round(2)
        
        log(f"Unique values: {df[target_col].nunique()}")
        log("")
        
        for val in target_dist.index:
            log(f"  {str(val):<20} {target_dist[val]:>6} ({target_pct[val]:.1f}%)")
    
    # =========================================================================
    # SECTION 7: OUTLIER SUMMARY
    # =========================================================================
    log("\n" + "─" * 70)
    log("7. OUTLIER SUMMARY (IQR Method)")
    log("─" * 70)
    
    outlier_summary = []
    
    for col in numeric_cols[:10]:  # Limit to 10 columns
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()
        
        if outlier_count > 0:
            outlier_summary.append({
                'Column': col,
                'Outliers': outlier_count,
                'Percentage': round(outlier_count / len(df) * 100, 2),
                'Normal_Range': f"[{lower:.2f}, {upper:.2f}]"
            })
    
    if outlier_summary:
        outlier_df = pd.DataFrame(outlier_summary)
        log(outlier_df.to_string(index=False))
    else:
        log("✅ No significant outliers detected")
    
    # =========================================================================
    # SECTION 8: CORRELATIONS (Top)
    # =========================================================================
    if len(numeric_cols) >= 2:
        log("\n" + "─" * 70)
        log("8. TOP CORRELATIONS")
        log("─" * 70)
        
        corr = df[numeric_cols].corr()
        
        # Get upper triangle
        upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        
        # Find top correlations
        top_corr = (upper_tri.stack()
                             .reset_index()
                             .rename(columns={0: 'Correlation', 
                                             'level_0': 'Variable_1', 
                                             'level_1': 'Variable_2'})
                             .assign(Abs_Corr=lambda x: x['Correlation'].abs())
                             .sort_values('Abs_Corr', ascending=False)
                             .head(10))
        
        for _, row in top_corr.iterrows():
            direction = "↑" if row['Correlation'] > 0 else "↓"
            log(f"  {row['Variable_1']:<20} ↔ {row['Variable_2']:<20} {direction} {row['Correlation']:.3f}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    log("\n" + "=" * 70)
    log("QUICK ACTIONS RECOMMENDED")
    log("=" * 70)
    
    recommendations = []
    
    if nulls.sum() > 0:
        recommendations.append(f"⚠️  Handle {nulls.sum()} missing values")
    if dups > 0:
        recommendations.append(f"⚠️  Remove {dups} duplicate rows")
    if outlier_summary:
        cols_with_outliers = len(outlier_summary)
        recommendations.append(f"⚠️  Investigate outliers in {cols_with_outliers} columns")
    if len(categorical_cols) > 0:
        recommendations.append(f"📌 Encode {len(categorical_cols)} categorical columns before modeling")
    
    if recommendations:
        for rec in recommendations:
            log(rec)
    else:
        log("✅ Data looks clean! Ready for analysis.")
    
    log("\n" + "=" * 70)
    
    # Save report
    if output_file:
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines))
        print(f"\n💾 Report saved to: {output_file}")
    
    return lines


def create_visualizations(df, target_col=None, save_plots=False):
    """
    Create standard EDA visualizations.
    
    Parameters:
    -----------
    df : pd.DataFrame
    target_col : str, optional
    save_plots : bool - Save plots to files
    """
    
    plt.style.use('seaborn-v0_8-darkgrid')
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target from feature lists
    if target_col:
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)
    
    # ─── Plot 1: Distributions of numeric columns ───
    if len(numeric_cols) > 0:
        n_cols = min(4, len(numeric_cols))
        n_rows = (len(numeric_cols[:8]) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
        
        for i, col in enumerate(numeric_cols[:8]):
            df[col].hist(bins=30, ax=axes[i], edgecolor='black', color='steelblue')
            axes[i].set_title(f'{col}', fontweight='bold')
            axes[i].set_ylabel('Frequency')
        
        for j in range(len(numeric_cols[:8]), len(axes)):
            axes[j].axis('off')
        
        plt.suptitle('Numeric Column Distributions', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('eda_distributions.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # ─── Plot 2: Categorical distributions ───
    if len(categorical_cols) > 0:
        n_cat = min(4, len(categorical_cols))
        fig, axes = plt.subplots(1, n_cat, figsize=(16, 4))
        axes = [axes] if n_cat == 1 else axes
        
        for i, col in enumerate(categorical_cols[:4]):
            df[col].value_counts().head(8).plot(kind='bar', ax=axes[i], color='coral')
            axes[i].set_title(f'{col}', fontweight='bold')
            axes[i].set_ylabel('Count')
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.suptitle('Categorical Column Distributions', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('eda_categorical.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # ─── Plot 3: Correlation heatmap ───
    if len(numeric_cols) >= 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        corr_cols = numeric_cols[:10]  # Limit to 10 for readability
        corr_matrix = df[corr_cols].corr()
        
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            ax=ax,
            square=True
        )
        
        ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('eda_correlation.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # ─── Plot 4: Missing values heatmap ───
    if df.isnull().sum().sum() > 0:
        fig, ax = plt.subplots(figsize=(12, 4))
        
        null_data = df.isnull()
        sns.heatmap(null_data.T, cmap='viridis', cbar=False, ax=ax)
        ax.set_title('Missing Values Map (yellow = missing)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Rows')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('eda_missing.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # ─── Plot 5: Target distribution ───
    if target_col and target_col in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        if df[target_col].dtype in ['object', 'category']:
            df[target_col].value_counts().plot(kind='bar', ax=ax, color='steelblue')
        else:
            df[target_col].hist(bins=20, ax=ax, color='steelblue')
        
        ax.set_title(f'Target Distribution: {target_col}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count')
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('eda_target.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    print("✅ Visualizations complete")


def main():
    """Main function for command-line usage."""
    
    parser = argparse.ArgumentParser(
        description='Quick EDA for any dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quick_eda.py data.csv
  python quick_eda.py data.csv --target Loan_Status
  python quick_eda.py data.csv --target Loan_Status --output report.txt
  python quick_eda.py data.csv --no-plots
        """
    )
    
    parser.add_argument('filepath', help='Path to CSV or Excel file')
    parser.add_argument('--target', help='Target column name', default=None)
    parser.add_argument('--output', help='Save report to file', default=None)
    parser.add_argument('--no-plots', action='store_true', help='Skip visualizations')
    parser.add_argument('--save-plots', action='store_true', help='Save plots as PNG files')
    
    args = parser.parse_args()
    
    # Load data
    df = load_data(args.filepath)
    
    # Run EDA
    run_quick_eda(df, target_col=args.target, output_file=args.output)
    
    # Visualizations
    if not args.no_plots:
        create_visualizations(df, target_col=args.target, save_plots=args.save_plots)


# =============================================================================
# USAGE EXAMPLES IN NOTEBOOK
# =============================================================================

def notebook_eda(df, target_col=None):
    """
    Simplified version for use in Jupyter/Colab notebooks.
    
    Usage:
        from scripts.quick_eda import notebook_eda
        notebook_eda(df, target_col='Loan_Status')
    """
    run_quick_eda(df, target_col=target_col)
    create_visualizations(df, target_col=target_col)


if __name__ == "__main__":
    # If run from command line
    if len(sys.argv) > 1:
        main()
    else:
        # Demo mode with sample data
        print("📊 Running demo with sample data...")
        print("Usage: python quick_eda.py your_file.csv --target target_column\n")
        
        np.random.seed(42)
        sample_df = pd.DataFrame({
            'age': np.random.randint(18, 70, 500),
            'income': np.random.gamma(4, 10000, 500),
            'loan_amount': np.random.gamma(3, 5000, 500),
            'category': np.random.choice(['A', 'B', 'C'], 500),
            'status': np.random.choice(['active', 'inactive'], 500),
            'approved': np.random.choice([0, 1], 500, p=[0.3, 0.7])
        })
        
        # Add some nulls
        sample_df.loc[np.random.choice(sample_df.index, 30), 'income'] = np.nan
        
        run_quick_eda(sample_df, target_col='approved')
        create_visualizations(sample_df, target_col='approved')
