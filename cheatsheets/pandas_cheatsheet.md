# 🐼 Pandas Cheat Sheet

## Loading Data

```python
pd.read_csv('file.csv')                 # CSV
pd.read_excel('file.xlsx')              # Excel
pd.read_sql(query, connection)          # SQL
pd.read_json('file.json')               # JSON
```

## Viewing Data

```python
df.head(n)                              # First n rows
df.tail(n)                              # Last n rows
df.sample(n)                            # Random n rows
df.info()                               # Column info
df.describe()                           # Statistics
df.shape                                # (rows, columns)
```

## Selection

```python
df['column']                            # Select column
df[['col1', 'col2']]                    # Multiple columns
df.loc[row, col]                        # By label
df.iloc[row, col]                       # By position
df[df['col'] > 5]                       # Filter rows
```

## Cleaning

```python
df.dropna()                             # Drop nulls
df.fillna(value)                        # Fill nulls
df.drop_duplicates()                    # Remove duplicates
df['col'].replace(old, new)             # Replace values
df.rename(columns={'old': 'new'})       # Rename
```

## Aggregation

```python
df.groupby('col').mean()                # Group and aggregate
df.groupby('col').agg(['mean', 'sum'])  # Multiple aggregations
df.pivot_table(values, index, columns)  # Pivot table
```

## Merging

```python
pd.concat([df1, df2])                   # Stack vertically
pd.merge(df1, df2, on='key')            # SQL-like join
```

## Creating Columns

```python
df['new'] = df['col1'] + df['col2']     # Math operations
df['new'] = df['col'].apply(func)       # Apply function
df['new'] = df['col'].map(dict)         # Map values
```

## Export

```python
df.to_csv('file.csv', index=False)      # To CSV
df.to_excel('file.xlsx', index=False)   # To Excel
df.to_sql('table', connection)          # To SQL
```
