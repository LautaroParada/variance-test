# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 10:32:33 2021

@author: lauta
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os
from eod import EodHistoricalData

from price_paths import PricePaths
from visuals import VRTVisuals
from variance_test import EMH

#%% Creating the reference variables

api_key = os.environ['API_EOD']
client = EodHistoricalData(api_key)

stock_prices = pd.DataFrame(
    client.get_prices_eod('SPY.US', period='d')
    )

#%% Testting the VRT
emh = EMH()                     # Initialization of the test class
q = 200
z, p = emh.vrt(X=stock_prices['close'].values, q=q, heteroskedastic=True)
