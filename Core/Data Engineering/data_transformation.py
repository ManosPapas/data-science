from standard import *


def days_difference(date1, date2):
    """Calculate the absolute difference in days between two dates."""
    return abs((date1 - date2).dt.days)


def create_intervals(data, interval_list, right=False):
    """Create intervals from data based on a list of intervals."""
    return pd.cut(data, interval_list, right=right)


def birthday_to_age(df):
    """Convert birth dates to ages in years."""
    return df["DOB"].apply(lambda x: (np.datetime64("today") - np.datetime64(x, "D")).astype("timedelta64[Y]").astype(int))


def safe_extract(url):
    """Safely extract domain components from a URL, returning an empty extraction if the URL is invalid."""
    if pd.isna(url) or not url:
        return tldextract.extract("")

    return tldextract.extract(url)
