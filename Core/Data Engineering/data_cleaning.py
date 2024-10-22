from references_calls import *


def keep_numbers(string):
    """Extract and return only the numeric part of a string."""
    numbers = re.sub("\D", "", str(string))
    return int(numbers)


def keep_text(string):
    """Extract and return only the alphabetic part of a string."""
    return re.sub("[^a-zA-Z]+", "", str(string))


def convert_money(df, local=True, charge_amount="Cost", currency_code="CurrencyCode", date_bought="CreatedAt", params={}):
    """Convert monetary values in a DataFrame based on exchange rates."""
    if params == {}:
        params = {
            "from_currency": "USD",
            "to_currency": "USD",
            "start_at" : "2021-01-01",
            "end_at": "9999-12-31 00:00:00"
        }

    rates = exchange_rates_version(params, local)
    df["ExchangeRate"] = 1.0
    for i, row in rates.iterrows():
        mask = (
            (df[currency_code] == row["FromCurrencyCode"]) &
            (df[date_bought] >= row["StartAt"]) & 
            (df[date_bought] <= row["EndAt"])
        ) & (df[currency_code] != params["to_currency"])
        df.loc[mask, "ExchangeRate"] = row["ConversionRate"]

    return df[charge_amount] * df["ExchangeRate"]


def exchange_rates_version(params, local=True):
    """Fetch currency exchange rates from a local or remote source."""
    if local:
        df = pd.read_csv(IMPORT_PATH + "References/currency_rate.csv")
        return currency_rate_types(df)
    else:    
        df = currency_rates_call(params)
        return currency_rate_types(df)


def apply_convert_money(amount, currency, rates, round_=2):
    """Apply the currency conversion to a given amount based on rates."""
    return round(amount * rates[currency], round_)


def get_latest_row(group, column="timestamp"):
    """Get the latest row in a DataFrame group based on a specified column."""
    return group.loc[group[[column]].max(axis=1).idxmax()]


def first_non_null(series):
    """Return the first non-null value in a Series."""
    return series.dropna().iloc[0] if not series.dropna().empty else None


def last_non_null(series):
    """Return the last non-null value in a Series."""
    return series.dropna().iloc[-1] if not series.dropna().empty else None


def drop_duplicate_columns(df):
    """Remove duplicate columns from a DataFrame."""
    df = df.loc[:, ~df.columns.duplicated()]
    return df
