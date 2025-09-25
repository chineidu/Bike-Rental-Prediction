# EDA Class Demo

This notebook demonstrates how to use the EDA class for comprehensive exploratory data analysis.

## Quick Start Example

```python
import pandas as pd
import numpy as np
from eda import EDA

# Create sample data
rng = np.random.default_rng(42)
data = pd.DataFrame({
    'age': rng.normal(35, 10, 1000),
    'salary': rng.exponential(50000, 1000),
    'score': rng.uniform(0, 100, 1000),
    'department': rng.choice(['Sales', 'Engineering', 'Marketing'], 1000),
    'experience': rng.choice(['Junior', 'Mid', 'Senior'], 1000, p=[0.4, 0.4, 0.2]),
    'target': rng.normal(75, 15, 1000)
})

# Initialize EDA
eda = EDA(data, target_column='target')

# Print summary
eda.print_summary()

# Get numeric statistics
numeric_stats = eda.numeric_summary()
print(numeric_stats)

# Get categorical statistics
cat_stats = eda.categorical_summary()
for col, stats in cat_stats.items():
    print(f"\\n{col}:")
    print(stats)

# Create visualizations
fig1 = eda.plot_numeric_distribution()
fig1.show()

fig2 = eda.plot_categorical_distribution()
fig2.show()

fig3 = eda.plot_correlation_heatmap()
fig3.show()

# Group analysis
group_stats = eda.group_analysis('department', ['age', 'salary'])
print(group_stats)

# Outlier detection
fig4 = eda.plot_outliers(method='iqr')
fig4.show()
```

## Key Features

### Numeric Data Analysis

- **Central Tendency**: Mean, median, mode
- **Spread**: Standard deviation, variance, range, IQR
- **Distribution Shape**: Skewness, kurtosis
- **Percentiles**: 25th, 50th, 75th percentiles
- **Outlier Detection**: IQR and Z-score methods
- **Missing Values**: Count and percentage

### Categorical Data Analysis

- **Frequency Analysis**: Counts and percentages
- **Mode Detection**: Most frequent category
- **Entropy Calculation**: Measure of distribution disorder
- **Missing Values**: Count and percentage

### Cross-Data Analysis

- **Correlation Analysis**: Pearson, Spearman, Kendall
- **Group-wise Statistics**: Numeric stats by categorical groups

### Visualizations (All Interactive with Plotly)

- **Numeric Distributions**: Histograms, box plots, violin plots
- **Categorical Distributions**: Bar charts, pie charts
- **Correlation Heatmaps**: Interactive correlation matrices
- **Outlier Detection Plots**: Visual outlier identification
- **Group Analysis Plots**: Box plots, violin plots by groups

## Usage with Real Data

```python
# Load your data
df = pd.read_csv('your_data.csv')

# Initialize EDA
eda = EDA(df, target_column='your_target_column')

# Generate comprehensive report
report = eda.generate_full_report()

# Create specific visualizations
fig = eda.plot_group_analysis('category_col', 'numeric_col', plot_type='violin')
fig.show()
```

## Advanced Features

- Automatic data type detection
- Robust handling of missing values
- Customizable outlier detection thresholds
- Support for large datasets
- Export-ready visualizations
- Comprehensive statistical summaries
