from helpers import *
from connect import *


def cancellations(params):
    df = execute_sql("Rudderstack/cancellations", DB_3, sql_params=params)
    return cancellations_types(df)


def cancellations_types(df):
    date_columns = ["cancelled_at"]
    return parse_dates_(df, date_columns)
