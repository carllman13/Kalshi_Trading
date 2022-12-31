#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import FedCurve
import time


# In[ ]:


#Set meetings
#need at least 2 meetings, in chronological order, of format 'FED-YYMMM' e.g.meetings = ['FED-23SEP','FED-23DEC']
#we have default_meetings = most recently settled contract + all unclosed meetings
meetings = FedCurve.default_meetings() 
meetings = ['FED-23SEP','FED-23NOV']

#Set lookback, default lookback =90
lookback = FedCurve.lookback
lookback = 10


# In[ ]:


#Get historical implied rate df - takes time
c = time.time()
meetings_implied_df = FedCurve.get_meetings_implied_df(series_ids = meetings,lookback = lookback)
d = time.time()
print(f'Process took {round((d-c)/60,2)}min')


# In[ ]:
print(lookback)

#Plotting dashboard
FedCurve.plot_historical_curve(meetings_implied_df)


# ## Checking specific data points

# e.g. We see an outlier on 23MAR Meeting on 22-10-21 from our dashboard, so we dig into the details by calling the check_meeting function:

# In[ ]:


historical_prices,contracts = FedCurve.check_meeting('23MAR',lookback = lookback)


# We then look at all strike contracts on the outlier date:

# In[ ]:


FedCurve.check_meeting_ondate(historical_prices,2022,10,21)


# - We see it's the '>3.25%' contract that's causing the anomolies (prices of lower strikes should >= higher strikes, since our CDF should be a non-decreasing step function)
# - So we pull up the detailed history for this specific strike over time

# In[ ]:


FedCurve.check_raw_data(contracts, '3.25%')


# We see that the price drop is due to sudden increase in spread,(bid/ask widened to 94) hence we couldn't have profited from this. 
