
def generate_data_profile(data):
    profile = {}

    profile['row_count'] = len(data)
    profile['column_count'] = len(data.columns)
    profile['memory_usage'] = f"{data.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB"
    dtypes = data.dtypes.value_counts()
    profile['data_types'] = {str(k): int(v) for k, v in dtypes.items()}
    profile['skewness'] = data.skew(numeric_only=True).to_dict()
    profile['kurtosis'] = data.kurtosis(numeric_only=True).to_dict()

    return profile
