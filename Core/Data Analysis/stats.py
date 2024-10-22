from standard import *
from constants import *


def sum_percentage(df_column, round_=2):
    """Calculate the percentage of a DataFrame column."""
    return round(df_column / df_column.sum() * 100, round_)


def count_percentage(df_column, round_=2):
    """Calculate the percentage of total counts in a DataFrame column."""
    return round(df_column / df_column.size * 100, round_)


def count_total(df_column):
    """Count the total number of entries in a DataFrame column."""
    return df_column.size


def sum_total(df_column, round_=0, data_type="int"):
    """Calculate the total sum of a DataFrame column."""
    return round(df_column.sum(), round_).astype(data_type)


def average_x_per_x(total_sum, total_count, round_=2):
    """Calculate the average value given a total sum and count."""
    return round(total_sum / total_count, round_)


def calculate_previous(current, previous):
    """Calculate the percentage change between current and previous values."""
    if (isinstance(current, pd.Series) and not isinstance(previous, pd.Series)) or (not isinstance(current, pd.Series) and isinstance(previous, pd.Series)):
        if isinstance(previous, pd.Series):
            previous = previous.iloc[0]
        if isinstance(current, pd.Series):
            current = current.iloc[0]

    try:
        if previous == 0:
            return 1

        result = round((current - previous) / previous, 3)
        
        if math.isinf(result):
            return 0
            
    except:
        return None
    
    return result
