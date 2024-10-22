from connect import *
from helpers import *


def products_call(params):
    df = execute_sql("Products/all_products", sql_params=params)
    return seat_preference_types(df)


def products_call_types(df):
    df = df.astype({
        "Cost"             : "float32",
        "width"            : "float32",
        "Colour"           : "string",
        "height"           : "float32",
        "CurrencyCode"     : "category"
    })

    date_columns = [
        "CreatedAt"
    ]

    return parse_dates_(df, date_columns)
