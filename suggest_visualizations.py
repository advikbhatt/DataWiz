# import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def suggest_visualizations(data):
    suggestions = []
    
    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    datetime_cols = data.select_dtypes(include=['datetime']).columns.tolist()
    
    for col in categorical_cols:
        unique_count = data[col].nunique()
        if unique_count > 1 and unique_count <= 20:
            suggestions.append({
                'type': 'bar',
                'columns': {'x': col},
                'title': f'Distribution of {col}'
            })
    
    for col in numeric_cols:
        suggestions.append({
            'type': 'histogram',
            'columns': {'x': col},
            'title': f'Distribution of {col}'
        })
    
    # Suggest scatter plots for pairs of numeric columns (limited to avoid too many)
    if len(numeric_cols) >= 2:
        for i in range(min(3, len(numeric_cols))):
            for j in range(i+1, min(4, len(numeric_cols))):
                suggestions.append({
                    'type': 'scatter',
                    'columns': {'x': numeric_cols[i], 'y': numeric_cols[j]},
                    'title': f'{numeric_cols[i]} vs {numeric_cols[j]}'
                })
    
    # Suggest pie charts for categorical columns with few categories
    for col in categorical_cols:
        unique_count = data[col].nunique()
        if unique_count > 1 and unique_count <= 8:
            suggestions.append({
                'type': 'pie',
                'columns': {'sector': col},
                'title': f'Distribution of {col}'
            })
    
    # Suggest boxplots for numeric columns, grouped by categorical columns
    if len(numeric_cols) > 0 and len(categorical_cols) > 0:
        for num_col in numeric_cols[:3]:  # Limit to first 3 numeric columns
            for cat_col in categorical_cols[:2]:  # Limit to first 2 categorical columns
                if data[cat_col].nunique() <= 10:  # Only for categorical with few values
                    suggestions.append({
                        'type': 'boxplot',
                        'columns': {'y': num_col, 'group': cat_col},
                        'title': f'{num_col} by {cat_col}'
                    })
    
    if len(numeric_cols) >= 3:
        suggestions.append({
            'type': 'correlation',
            'columns': {},
            'title': 'Correlation Matrix'
        })
    
    return suggestions
