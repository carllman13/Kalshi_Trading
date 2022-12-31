#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from datetime import datetime


# ## Username and Password

# In[1]:


username = '' #your username
password = '' #your password


# ## Other parameters

# In[ ]:


#Manually adjust historical FOMCs
#Date is set to be actual release date + 1
#e.g. 22Dec meeting took place on Dec 14th, 2022, so We set the confirmed rate to be updated on Dec 15th, 2022.
actual_FOMC={
    '22SEP':[datetime(2022,9,22),300],
    '22NOV':[datetime(2022,11,3),375],
    '22DEC': [datetime(2022,12,15),425],
    }
actual_FOMC_df = pd.DataFrame.from_dict(actual_FOMC,orient= 'index',columns = ['Update Date','FFR in bps'])

#default historical window is 90
lookback = 90

