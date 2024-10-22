from helpers import *
from connect import *


def currency_rates_call(params):
    df = execute_sql("References/currency_rates", db_name=DB_2, sql_params=params)
    return currency_rate_types(df)


def currency_rate_types(df):
    date_columns = [
        "StartAt",
        "EndAt"
    ]

    return parse_dates_(df, date_columns)


def latest_currency_rates_call(params):
    df = execute_sql("References/latest_currency_rates", db_name=DB_2, sql_params=params)
    return currency_rate_types(df)


def latest_currency_rates_types(df):
    date_columns = [
        "StartAt"
    ]

    return parse_dates_(df, date_columns)