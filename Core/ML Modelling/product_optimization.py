from main import *
from standard import *


def product_desirability(df, total_routes):
    """Calculate product desirability for each row in the DataFrame."""
    return df.apply(lambda row: apply_product_desirability(row, total_routes), axis=1)


def apply_product_desirability(row, total_routes, purchases_weight=1, picked_order_weight=1):
    """Calculate desirability score for a single row based on purchases."""

    desirability = round((purchases_weight * row["Purchases"] +
            picked_order_weight * row["Purchases"] - math.log1p(row["DaysAfterPurchase"])) / total_routes)
    
    return desirability


def product_desirability_bucket(df):
    """Transform desirability scores using the log1p function."""
    return round(df["Desirability"].apply(np.log1p))


def product_picked_order(df):
    """Calculate and assign picked order numbers for each travel class in the DataFrame."""
    if "PickedOrder" not in df.columns:
        df["PickedOrder"] = 0

    df.sort_values(by=["InventoryID", "CreatedAt"], inplace=True)
    economy_mask = df["Code"].isin(ECONOMY_PRODUCT_CLASSES)
    df.loc[economy_mask, "PickedOrder"] = df[economy_mask].groupby("InventoryID").cumcount() + 1
    discount = df["Code"].isin(DISCOUNT_CODE)
    df.loc[discount, "PickedOrder"] = df[discount].groupby("InventoryID").cumcount() + 1
    return df["PickedOrder"]
