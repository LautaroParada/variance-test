# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 10:32:33 2021

@author: lauta
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from datetime import datetime
import os
from polygon import RESTClient

from price_paths import PricePaths
from visuals import VRTVisuals
from variance_test import EMH

#%% Creating the reference variables

api_key = os.environ['API_POLY']
client = RESTClient(api_key)

base = 'XRP'
quote = 'USD'
date_ = datetime.today().strftime("%Y-%m-%d")

#%% Requesting the data for the selected instruments

def tick_extractor(req):
    """
    Convert the request into a structured pandas df

    Parameters
    ----------
    req : model responso from Polygon

    Returns
    -------
    pd.Dataframe

    """

    return pd.DataFrame(
        data={
            'prices':       [ req.ticks[i]['p'] for i in range(len(req.ticks)) ],
            'sizes':        [ req.ticks[i]['s'] for i in range(len(req.ticks)) ],
            'timestamp':    [ req.ticks[i]['t'] for i in range(len(req.ticks)) ],
            'exchange':     [ req.ticks[i]['x'] for i in range(len(req.ticks)) ],
            'conditions':   [ req.ticks[i]['c'] for i in range(len(req.ticks)) ]
            }
        )

resp = client.crypto_historic_crypto_trades(base, quote, date_, limit=1000)
test = tick_extractor(req=resp)
plt.plot(test.iloc[:, 0])

plt.show()

#%% Testting the VRT
emh = EMH()                     # Initialization of the test class
q = 5
z, p = emh.vrt(X=test.iloc[:, 0].values, q=q, heteroskedastic=True)
#%%
resp = client.stocks_equities_aggregates('X:BTCUSD', 3, 'minute', 
                                         from_='2020-10-14', to='2021-02-14', limit=50000)