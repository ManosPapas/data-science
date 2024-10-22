import os
import yaml

# Global Paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")) + "/"
CORE_DIR = ROOT_DIR + "Core/"
IMPORT_PATH = ROOT_DIR + "xTemp/xData/imports/"
EXPORT_PATH = ROOT_DIR + "xTemp/xData/exports/"
EXPORT_POWERBI_PATH = ROOT_DIR + "xTemp/xData/exports/_PowerBI/"
SQL_PATH = ROOT_DIR + "SQL/"

# Config variables
with open(ROOT_DIR + "config.yml", "r") as file:
    CONFIG = yaml.safe_load(file)

# Databases' Names
DB_1 = "db1"
DB_2 = "db2"
DB_3 = "db3"

# Currencies
CURRENCIES = {"NOK", "EUR", "GBP", "USD"}

DISCOUNT_CODE = {"Disc1", "Disc2", "Disc3"}

# Age and Days Intervals
AGE_INTERVALS = [0, 3, 5, 9, 12, 16, 18, 25, 36, 46, 56, 500]
DAYS_INTERVALS = [0, 2, 31, 91, 365 * 5]
