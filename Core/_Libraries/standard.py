import functools
from functools import reduce

import math
import operator
import os
import random
import re
import statistics
import time
import warnings
from datetime import datetime as dt
from datetime import timedelta

import numpy as np
import pandas as pd
pd.options.display.float_format = '{:_.2f}'.format

import scipy.stats as stats
import statsmodels.api as sm
import tldextract

# Suppress warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
