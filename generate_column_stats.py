import pandas as pd
def generate_column_stats(data, column_name):
    stats = {}
    column = data[column_name]
    
    stats['missing_count'] = int(column.isna().sum())
    stats['missing_percent'] = float((stats['missing_count'] / len(data)) * 100)
    stats['unique_values'] = int(column.nunique())
    
    if pd.api.types.is_numeric_dtype(column):
        stats['type'] = 'numeric'
        valid_data = column.dropna()
        if len(valid_data) > 0:
            stats['min'] = float(valid_data.min())
            stats['max'] = float(valid_data.max())
            stats['mean'] = float(valid_data.mean())
            stats['median'] = float(valid_data.median())
            stats['std'] = float(valid_data.std())
            stats['q1'] = float(valid_data.quantile(0.25))
            stats['q3'] = float(valid_data.quantile(0.75))
            stats['zero_count'] = int((valid_data == 0).sum())
            stats['negative_count'] = int((valid_data < 0).sum())
            if len(valid_data) > 2: 
                stats['skewness'] = float(valid_data.skew())
                stats['kurtosis'] = float(valid_data.kurtosis())
        else:
            stats['min'] = stats['max'] = stats['mean'] = stats['median'] = stats['std'] = None
            stats['q1'] = stats['q3'] = stats['zero_count'] = stats['negative_count'] = None
            stats['skewness'] = stats['kurtosis'] = None
    
    elif pd.api.types.is_bool_dtype(column):
        stats['type'] = 'boolean'
        valid_data = column.dropna()
        if len(valid_data) > 0:
            true_count = int(valid_data.sum())
            false_count = int(len(valid_data) - true_count)
            stats['true_count'] = true_count
            stats['false_count'] = false_count
            stats['true_percent'] = float((true_count / len(valid_data)) * 100)
            stats['false_percent'] = float((false_count / len(valid_data)) * 100)
        else:
            stats['true_count'] = stats['false_count'] = 0
            stats['true_percent'] = stats['false_percent'] = 0
    
    elif pd.api.types.is_datetime64_any_dtype(column):
        stats['type'] = 'datetime'
        valid_data = column.dropna()
        if len(valid_data) > 0:
            stats['min'] = valid_data.min().isoformat()
            stats['max'] = valid_data.max().isoformat()
            stats['range_days'] = (valid_data.max() - valid_data.min()).days
        else:
            stats['min'] = stats['max'] = None
            stats['range_days'] = None
    
    else:
        stats['type'] = 'categorical'
        valid_data = column.dropna()
        if len(valid_data) > 0:
            value_counts = valid_data.value_counts()
            if len(value_counts) > 0:
                stats['top_value'] = str(value_counts.index[0])
                stats['top_count'] = int(value_counts.iloc[0])
                stats['top_percent'] = float((value_counts.iloc[0] / len(valid_data)) * 100)
            else:
                stats['top_value'] = stats['top_count'] = stats['top_percent'] = None
            
            top_values = value_counts.head(5).reset_index()
            top_values.columns = ['value', 'count']
            stats['top_values'] = [
                {'value': str(row['value']), 'count': int(row['count'])} 
                for _, row in top_values.iterrows()
            ]
        else:
            stats['top_value'] = stats['top_count'] = stats['top_percent'] = None
            stats['top_values'] = []
    
    return stats


