import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

def create_visualization(data, chart_type, selected_columns, color='#4a56a6', filters={}, max_rows=20):
    filtered_data = data.copy()
    for col, val in filters.items():
        if val is not None:
            filtered_data = filtered_data[filtered_data[col] == val]

    plot_data = filtered_data.head(max_rows)

    if chart_type == 'bar':
        col, group_col = selected_columns.get('x'), selected_columns.get('group', None)
        if col not in data.columns:
            return None
        valid_data = plot_data[~plot_data[col].isna()]
        if valid_data.empty:
            return None
        if group_col and group_col in data.columns:
            fig = px.bar(valid_data, x=col, color=group_col, barmode='group',
                         title=f'Distribution of {col} grouped by {group_col}', color_discrete_sequence=px.colors.qualitative.Vivid)
        else:
            value_counts = valid_data[col].value_counts().reset_index()
            value_counts.columns = ['value', 'count']
            fig = px.bar(value_counts, x='value', y='count', labels={'value': col, 'count': 'Count'},
                         title=f'Distribution of {col}', color='value', color_discrete_sequence=px.colors.qualitative.Vivid)

    elif chart_type == 'histogram':
        col = selected_columns.get('x')
        if col not in data.columns or not pd.api.types.is_numeric_dtype(data[col]):
            return None
        valid_data = plot_data[~plot_data[col].isna()]
        if valid_data.empty:
            return None
        fig = px.histogram(valid_data, x=col, title=f'Histogram of {col}',
                           color_discrete_sequence=px.colors.qualitative.Pastel, marginal='box')

    elif chart_type == 'pie':
        col = selected_columns.get('sector')
        if col not in data.columns:
            return None
        valid_data = plot_data[~plot_data[col].isna()]
        if valid_data.empty:
            return None
        value_counts = valid_data[col].value_counts().reset_index()
        value_counts.columns = ['value', 'count']
        fig = px.pie(value_counts, values='count', names='value', title=f'Distribution of {col}',
                     color_discrete_sequence=px.colors.qualitative.Set3)

    elif chart_type == 'scatter':
        x_col, y_col, group_col = selected_columns.get('x'), selected_columns.get('y'), selected_columns.get('group', None)
        if x_col not in data.columns or y_col not in data.columns or not pd.api.types.is_numeric_dtype(data[x_col]) or not pd.api.types.is_numeric_dtype(data[y_col]):
            return None
        valid_data = plot_data[~(plot_data[x_col].isna() | plot_data[y_col].isna())]
        if valid_data.empty:
            return None
        fig = px.scatter(valid_data, x=x_col, y=y_col, color=group_col, title=f'{x_col} vs {y_col}',
                         opacity=0.7, color_discrete_sequence=px.colors.qualitative.Plotly)

    elif chart_type == 'boxplot':
        col, group_col = selected_columns.get('y'), selected_columns.get('group', None)
        if col not in data.columns or not pd.api.types.is_numeric_dtype(data[col]):
            return None
        valid_data = plot_data[~plot_data[col].isna()]
        if group_col and group_col in data.columns:
            valid_data = valid_data[~valid_data[group_col].isna()]
            fig = px.box(valid_data, y=col, x=group_col, title=f'Boxplot of {col} by {group_col}',
                         color=group_col, color_discrete_sequence=px.colors.qualitative.Plotly)
        else:
            fig = px.box(valid_data, y=col, title=f'Boxplot of {col}', color_discrete_sequence=[color])

    elif chart_type == 'correlation':
        numeric_cols = plot_data.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) < 2:
            return None
        corr_matrix = plot_data[numeric_cols].corr().round(2)
        fig = px.imshow(corr_matrix, text_auto=True,
                        color_continuous_scale=px.colors.diverging.RdBu,
                        title='Correlation Matrix')

    elif chart_type == 'line':
        x_col, y_col, group_col = selected_columns.get('x'), selected_columns.get('y'), selected_columns.get('group', None)
        if x_col not in data.columns or y_col not in data.columns:
            return None
        fig = px.line(plot_data, x=x_col, y=y_col, color=group_col, title=f'Line Chart: {x_col} vs {y_col}',
                      color_discrete_sequence=px.colors.qualitative.Plotly)

    elif chart_type == 'area':
        x_col, y_col = selected_columns.get('x'), selected_columns.get('y')
        if x_col not in data.columns or y_col not in data.columns:
            return None
        fig = px.area(plot_data, x=x_col, y=y_col, title=f'Area Chart: {x_col} vs {y_col}', color_discrete_sequence=[color])

    elif chart_type == 'violin':
        col, group_col = selected_columns.get('y'), selected_columns.get('group', None)
        if col not in data.columns or not pd.api.types.is_numeric_dtype(data[col]):
            return None
        fig = px.violin(plot_data, y=col, color=group_col, title=f'Violin Plot of {col}', color_discrete_sequence=px.colors.qualitative.Plotly)

    else:
        return None

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f0f2f6'),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig
