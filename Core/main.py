# Load everything. 
# Use this file to load all functionality.
import os
import sys

# Do not create cash files.
sys.dont_write_bytecode = True

# Global Path
MAIN_PATH = os.path.dirname(os.path.abspath(__file__)) + '/'

# Load directories.
sys.path.insert(0, os.path.dirname(MAIN_PATH))
sys.path.insert(0, os.path.dirname(MAIN_PATH + '_Global/'))
sys.path.insert(0, os.path.dirname(MAIN_PATH + '_SQL Calls/'))
sys.path.insert(0, os.path.dirname(MAIN_PATH + '_SQL Calls/Commerce/'))
sys.path.insert(0, os.path.dirname(MAIN_PATH + '_SQL Calls/Partners/'))
sys.path.insert(0, os.path.dirname(MAIN_PATH + '_Libraries/'))
sys.path.insert(0, os.path.dirname(MAIN_PATH + 'Data Analysis/'))
sys.path.insert(0, os.path.dirname(MAIN_PATH + 'Data Engineering/'))
sys.path.insert(0, os.path.dirname(MAIN_PATH + 'ML Modelling/'))
sys.path.insert(0, os.path.dirname(MAIN_PATH + 'Unit Tests/'))

# Load files.
# Load Libraries.
from standard import *
from connection import *
from plot import *
from model import *

# Load Global.
from constants import *
from helpers import *
from connect import *

# Load SQL Calls.
from customers import *
from products import *
from references_calls import *
from orders import *
from cancellations import *

# Load Data Analysis.
from stats import *

# Load Data Engineering.
from data_cleaning import *
from data_transformation import *

# Load ML Modelling.
from product_optimization import *
