from helpers import *
from connect import *


def orders_call(params):
    df = execute_sql("Rudderstack/orders", DB_3, sql_params=params)
    return orders_types(df)


def orders_types(df):
    df = df.astype({
        "order_type"     : "category",
        "currency_code"  : "category",
        "total_cost"     : "float32"
    })

    date_columns = ["booking_time"]
    return parse_dates_(df, date_columns)
