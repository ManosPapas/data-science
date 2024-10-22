from helpers import *
from connect import *


def total_customers_call(params):
    return execute_sql("Customers/total_customers", sql_params=params).iloc[0, 0]


def customer_details(params):
    df = execute_sql("Customers/customer_details", db_name=DB_1, sql_params=params)
    return customer_details_types(df)


def customer_details_types(df):
    date_columns = [
        "CreatedAt",
        "Birthday",
        "LastLoginAt"
    ]

    return parse_dates_(df, date_columns)
