def handle_missing_values(data):
    cleaned_data = data.copy()
    missing_info = []

    for column in cleaned_data.columns:
        missing_count = cleaned_data[column].isna().sum()
        missing_percent = (missing_count / len(cleaned_data)) * 100
        
        if missing_count > 0:
            missing_info.append({
                'column': column,
                'missing_count': missing_count,
                'missing_percent': missing_percent
            })
    
    return cleaned_data, missing_info