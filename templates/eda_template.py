"""
Quick EDA Template
Author: Fredys Caballero
Version: 1.0

Usage:
    from templates.eda_template import explore_data
    summary = explore_data(df, target_col='target')
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def explore_data(df, target_col=None, show_plots=True):
    """
    Complete exploratory data analysis in one function
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to explore
    target_col : str, optional
        Name of target/label column
    show_plots : bool, default=True
        Whether to display visualizations
    
    Returns:
    --------
    dict : Summary statistics and findings
    """
    
    print("="*70)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    # 1. Dimensions
    print(f"\n📊 DIMENSIONS: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # 2. Data types
    print(f"\n📋 DATA TYPES:")
    print(df.dtypes.value_counts())
    
    # 3. Missing values
    print(f"\n⚠️  MISSING VALUES:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    missing_df = pd.DataFrame({
        'Count': missing,
        'Percentage': missing_pct
    }).query('Count > 0').sort_values('Count', ascending=False)
    
    if len(missing_df) > 0:
        print(missing_df)
    else:
        print("✅ No missing values")
    
    # 4. Duplicates
    duplicates = df.duplicated().sum()
    print(f"\n🔄 DUPLICATES: {duplicates} rows ({duplicates/len(df)*100:.2f}%)")
    
    # 5. Summary statistics
    print(f"\n📈 SUMMARY STATISTICS:")
    print(df.describe())
    
    # 6. Target distribution (if provided)
    if target_col and target_col in df.columns:
        print(f"\n🎯 TARGET DISTRIBUTION ({target_col}):")
        print(df[target_col].value_counts())
        print("\nPercentages:")
        print((df[target_col].value_counts(normalize=True) * 100).round(2))
    
    # 7. Visualizations
    if show_plots:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Create subplots
        n_plots = min(4, len(numeric_cols) + len(categorical_cols))
        if n_plots > 0:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()
            
            plot_idx = 0
            
            # Numeric distributions
            for col in numeric_cols[:2]:
                if plot_idx < 4:
                    df[col].hist(bins=30, ax=axes[plot_idx], edgecolor='black')
                    axes[plot_idx].set_title(f'Distribution: {col}')
                    axes[plot_idx].set_ylabel('Frequency')
                    plot_idx += 1
            
            # Categorical distributions
            for col in categorical_cols[:2]:
                if plot_idx < 4:
                    df[col].value_counts().plot(kind='bar', ax=axes[plot_idx])
                    axes[plot_idx].set_title(f'Distribution: {col}')
                    axes[plot_idx].set_ylabel('Count')
                    axes[plot_idx].tick_params(axis='x', rotation=45)
                    plot_idx += 1
            
            # Hide unused subplots
            for idx in range(plot_idx, 4):
                axes[idx].axis('off')
            
            plt.tight_layout()
            plt.show()
        
        # Correlation matrix
        if len(numeric_cols) >= 2:
            plt.figure(figsize=(10, 8))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Matrix')
            plt.tight_layout()
            plt.show()
    
    # Return summary
    summary = {
        'rows': df.shape[0],
        'columns': df.shape[1],
        'missing_values': missing.sum(),
        'duplicates': duplicates,
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(df.select_dtypes(include=['object']).columns),
        'memory_mb': df.memory_usage(deep=True).sum() / 1024**2
    }
    
    print(f"\n{'='*70}")
    print("✅ EXPLORATION COMPLETED")
    print(f"{'='*70}\n")
    
    return summary


def quick_eda(df):
    """Ultra-fast EDA - 5 essential checks only"""
    print(f"Shape: {df.shape}")
    print(f"\nNulls:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nTypes:\n{df.dtypes.value_counts()}")
    print(f"\nStats:\n{df.describe()}")
    
    # Quick visualization
    numeric_col = df.select_dtypes(include=['number']).columns
    if len(numeric_col) > 0:
        df[numeric_col[0]].hist(bins=30)
        plt.title(f'Distribution: {numeric_col[0]}')
        plt.show()


# Example usage
if __name__ == "__main__":
    # Test with sample data
    sample_df = pd.DataFrame({
        'age': np.random.randint(18, 70, 1000),
        'income': np.random.gamma(4, 10000, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000),
        'target': np.random.choice([0, 1], 1000)
    })
    
    summary = explore_data(sample_df, target_col='target')
    print("\nSummary:", summary)
