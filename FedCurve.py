#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Installing packages
#%pip install panel --user
#pip install panel==0.12 --user
#!pip install hvplot==0.7.3 --user
#conda install -c pyviz panel 
#conda install -c pyviz hvplot
#conda install -c conda-forge hvplot


# In[4]:


import os
import urllib
from KalshiClientsBaseV2 import ExchangeClient
import pandas as pd
import numpy as np
from datetime import datetime
import json
import time
import datetime as dt
import uuid
import urllib.request,urllib.parse,urllib.error
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from matplotlib.pyplot import figure, xticks
from ipywidgets import widgets, interact, interactive, fixed, interact_manual,GridspecLayout
from scipy.interpolate import CubicSpline
import ipywidgets as widgets
import User


# In[5]:


username = User.username
password = User.password
demo_api_base = 'https://trading-api.kalshi.com/trade-api/v2'

exchange_client = ExchangeClient(demo_api_base, username, password)
print(exchange_client.get_exchange_status())

lookback = User.lookback


# ## Functions for retrieving data

# In[6]:


def get_content(tickr, params, func, content_name,max_pages = None):
    '''
    returns all history cleaned dataframe
    content_name is the desired string from the json output (either 'history' or 'trades')
    sometimes people want to limit max_pages to avoid getting too many data
    '''
    
    histories = []
    content = func(**params)
    histories.append(pd.DataFrame(content[content_name]))
    params['cursor'] = content['cursor']

    if max_pages is None:
        while content['cursor'] is not None and content['cursor'] !='':
            content = func(**params)
            histories.append(pd.DataFrame(content[content_name]))
            params['cursor'] = content['cursor']
    # if max_pages is specified, then I only retrieve at most max_pages pages
    else:
        for i in tqdm(range(max_pages-1)):
            content = func(**params)
            histories.append(pd.DataFrame(content[content_name]))
            params['cursor'] = content['cursor']
            if content['cursor'] is None or content['cursor'] =='':
                break
                
    histories_df = pd.concat(histories)
    try: #some calls may contain a 'ts' column - we convert that into normal datetime
        histories_df['Date']= histories_df['ts'].apply(lambda x: datetime.fromtimestamp(x).date())
    except:
        p = 1 #not do anything

    return histories_df

def contract_history(tickr, min_ts = None, max_ts = None,max_pages = None):
    '''
    Returns all contract history cleaned dataframe without duplicates.
    Analogous to exchange_client.get_market_history()
    '''
    params =     {
                         'limit': 100, #100 is the max limit per page
                         'min_ts' : min_ts,
                         'max_ts':max_ts,
                        'cursor':None,
    'ticker':tickr}

    contract_histories_df = get_content(tickr = tickr, params = params, func = exchange_client.get_market_history, content_name = 'history',max_pages = max_pages)
    
    return contract_histories_df

def trade_history(tickr, min_ts = None, max_ts = None,max_pages=None):
    '''
    Returns all trade history cleaned dataframe 
    Analogous to exchange_client.get_trades()
    Outputs df columns: yesprice,yesbid,yesask,nobid,noask, volume,openinterest,ts,Time
    '''
    params =     {
                         'limit': 100, #100 is the max limit per page
                         'min_ts' : min_ts,
                         'max_ts':max_ts,
                        'cursor':None,
    'ticker':tickr}
    trade_histories_df = get_content(tickr = tickr, params = params, func = exchange_client.get_trades, content_name = 'trades',max_pages=max_pages)
    
    return trade_histories_df

def specified_markets(min_close_ts = None, max_close_ts = None,event_ticker = None,series_ticker = None,tickers = None, status = None):
    '''
    Specify some parameters (time/event/series/status),returns a df of all related contracts and related info.
    Analogous to exchange_client.get_markets()
    '''
    markets_params =     {
                         'limit': 100, #100 is the max limit per page
                         'min_close_ts' : min_close_ts,
                         'max_close_ts':max_close_ts,
                         'event_ticker':event_ticker,
                         'series_ticker':series_ticker,
                         'cursor':None,
                         'tickers':tickers,
                         'status' : status}
    markets = []
    market = exchange_client.get_markets(**markets_params)
    markets.append(pd.DataFrame(market['markets']))
    markets_params['cursor'] = market['cursor']

    while market['cursor'] !='' and market['cursor'] is not None:
        market = exchange_client.get_markets(**markets_params)
        markets.append(pd.DataFrame(market['markets']))
        markets_params['cursor'] = market['cursor']
        #print(markets_params['cursor'])
    
    markets_df= pd.concat(markets)
    return markets_df
    


# ## Functions for setting up default meetings

# In[7]:


def get_recent_close(actual_FOMC):
    '''
    Retuns the latest meeting and rate as a tuple e.g.('22DEC',425) 
    '''
    max_date = datetime(2010,1,1)
    for key,value in actual_FOMC.items():
        if value[0]>max_date:
            max_date = value[0]
            meeting = key
            bps = value[1]
    return (meeting,bps)

def unsettled_contracts(contract_type = 'FED'):
    '''
    Function for getting all unsettled contract IDs about that event
    '''
    df = specified_markets(event_ticker = contract_type,status = 'open')
    #filitering to make sure we only take our desired contracts with series ID like 'FED-23FEB'
    filtered_df = df[(df['event_ticker'].str[:4]=='FED-') & (df['event_ticker'].str[4:6].str.isnumeric()) & (df['event_ticker'].str.len()==9)]
    #display in timeorder e.g. ['23FEB','23MAR'...]
    contracts = list(filtered_df.event_ticker.unique()) 
    contracts.reverse() 
    
    return contracts


# ## Functions for working with df of historical implied rates for different meetings

# Workflow: build_df -> build_df_each_rate -> cum_df  -> discrete_df -> meetings_implied_df

# In[8]:


def build_df(contract_id, category,lookback):
    '''
    Function that returns daily closing price of that contract in a dataframe
    Category refers to the type/strike e.g."5.00%"
    Requires a lookback to specify earliest time otherwise the function will collect too much data since the launch of the contract 
    Output has index Time, columns [volume, open_interest, price, spread] 
    Example of calling the function: build_df(contract_id = 'FED-23MAY-T5.00',category = '5.00',lookback = 30) 
    '''
    
    #Current date
    current_time = datetime.today()
    current_date = current_time.date()
    
    start_time = int((current_time - dt.timedelta(days=lookback)).timestamp()) 
    
    df = contract_history(tickr = contract_id, min_ts = start_time)
    #original df has columns: yesprice,yesbid,yesask,nobid,noask, volume,openinterest,ts,Time
    df['price'] = (df['yes_bid'] + df['yes_ask'])/2
    df['spread'] = df['yes_ask'] - df['yes_bid']
    df = df.drop(['yes_bid','yes_ask','yes_price','no_bid','no_ask'],axis = 1)
    
    #Get only daily closing price with date column. There may be multiple ts a day. only use the last one.
    df['Month/Day'] = df['Date'].apply(lambda x: str(x.month)+'/' + str(x.day))
    df['closing'] = np.where(df['Month/Day']!=df['Month/Day'].shift(-1),1,0)
    df.iloc[-1,df.columns.get_loc('closing')] = 1  #filling the last timestamp also to 1
    df_target =df[df.closing == 1]
    df_target = df_target.reset_index(drop = True)
    
    try:
    #add in today's date in case there is no observation today
        last_date = df_target['Date'][df_target.shape[0]-1]
        if last_date != current_date:
            df_target = df_target.append({'Date':current_date},ignore_index=True)  
    except:
        #we get an empty data frame, so do nothing (error occurs because df_target['Time'] has no length#
        1
        
    df_target['Category'] = category
    df_target = df_target.drop(['Month/Day','closing','ts'],axis = 1)

    ##autofill missing date value same as last observed value, making sure there's data on every day
    r = pd.date_range(start=df_target.Date.min(), end=df_target.Date.max())
    df_target = df_target.set_index('Date').reindex(r).fillna(method = 'ffill').rename_axis('Date').reset_index()
    
    return df_target


# In[9]:


def to_bps(string):
    return float(string.strip('%'))*100

def build_df_each_rate(contract_ids,categories,lookback):
    '''
    For one specific meeting, takes in a list of ids and list of categories(i.e. 5.00%)
    Applies build_df
    Return a dictionary of dfs, each df contains daily info of that specific contract
    '''
    
    contracts = {}
    for contract_id,category in zip(contract_ids,categories):
        contracts[category] = build_df(contract_id,category,lookback = lookback)
    return contracts

def cumulative_df(contracts):
    '''
    takes in dic of contracts df for specific meeting, (output of build_df_each_rate)
    returns a df representing cumulative possibilities across time
    Each row is a date, each column shows the price for the dif rates
    month,year allows us to manually adjust for erroneous data
    '''
    
    #Current date
    current_time = datetime.today()
    current_date = current_time.date()
    
    #initiate an empty df with date index
    final_df = pd.DataFrame({'Date': []})
    categories = list(contracts.keys())  #list of ['>3.00%, >3.25%', etc]
   
    #Sometimes different contracts in the same series will open at different times
    #we therefore try to use the date of the earliest contract as benchmark for our date index
    early_date = current_date
    earliest_contract = categories[0] 
    for key,df in contracts.items():
        if df.loc[0].Date < early_date:
            early_date = df.loc[0].Date #updating the earliest date 
            earliest_contract = key  #updating the earliest contract
    final_df['Date'] = contracts[earliest_contract].Date   #assumed we used the date available for the contract date
    
    #joining the different contracts for that series on Date
    for bp,df in contracts.items():
        final_df = pd.merge(final_df,df[['Date','price']],on = 'Date',how='left')
        bp = bp[1:6]   #Some contracts are titled '>4.25% :: +25bp hike' etc. We only want the the numbers
        final_df = final_df.rename(columns = {'price':bp} )
        
    #missing values are filled with 0 - unable to trade
    final_df = final_df.interpolate().fillna(0).set_index('Date').div(100)
    
    return final_df


# In[10]:


def discrete_df(cumulative_df,month,year):
    '''
    Create new df for discrete rather than cumulative probabilities
    Output is a df with date as index and 1 column of implied rates on that date
    '''
    discrete_df = cumulative_df.copy()
        
    for ind, column in enumerate(cumulative_df.columns):
        #calculates the discrete probabilities by subtracting adjacent cumulative probabilities
        if ind < len(cumulative_df.columns)-1: 
            discrete_df[column] = cumulative_df[column] - cumulative_df.iloc[:,ind+1]
            
    rates=[]
    for rate in cumulative_df.columns:
        rates.append(to_bps(rate))

    def get_weighted(values):
        '''
        Function for calculating weighted avg
        '''
        #one series my have 0 price for lower, less liquid strikes
        #which leads to negative values in our discrete probabilities.
        #hence when calculating the weighted avg, we convert these negatives to 0
        values = [max(0,val) for val in values ]
        try: 
            avg = np.average(a = rates,weights = values)
        except:
            avg = 0
        return avg

    #Calculate implied rates with weighted average
    discrete_df['Implied_rate'] = discrete_df.apply(get_weighted,axis = 1).round(2)
    kalshi_df = discrete_df['Implied_rate'].to_frame(name = f'{year}{month}')
    
    actual_FOMC_df = User.actual_FOMC_df
    
    #updating settled markets on realized Fed Funds rates
    for meeting in kalshi_df.columns:
        #the actual FFR release date: Real World Meeting date +1 (e.g. Rate announced on Nov 14th, 
        #update will start on Nov 15th)
        try:
            update_date = actual_FOMC_df.loc[meeting,'Update Date']   
            #updating any dates on/after that date
            kalshi_df.loc[update_date:,meeting] = actual_FOMC_df.loc[meeting,'FFR in bps']
        except KeyError: #this market hasn't settled yet (i.e. we don't have the meeting in the actual_FOMC_df data frame)
            pass  #we don't do anything
        
    return kalshi_df


# In[11]:


def get_meetings_implied_df(series_ids,lookback):
    '''
    Function for getting historical implied rates daily for different meetings in my [series ids] list
    e.g. series_ids = ['FED-22DEC','FED-23FEB','FED-23MAR']
    Returns a df with each row being implied rates on that date, and each column being the different meetings
    '''
    
    series_dfs = {}
    for series_id in series_ids:
        #get df for each contract in the series in order to retrieve all tickers for that series
        series_content_df = specified_markets(series_ticker= series_id)
        #for each series, get dictionary of df for specific contracts
        try:
            contracts = build_df_each_rate(contract_ids = series_content_df.ticker.tolist(), categories = series_content_df.subtitle.tolist(),lookback = lookback) 
        except:
            raise ValueError(f"each element in series_ids need to be formatted like: 'FED-YYMMM', or need longer lookback period")

        #we put the dictionary into another dictionary
        series_dfs[series_id] = contracts

    meetings = {}
    for meeting,historical_rates in series_dfs.items():
        cum_df = cumulative_df(historical_rates).iloc[:,::-1]
        imp_rates = discrete_df(cum_df,month = meeting[-3:], year =meeting[-5:-3]) #naming the columns
        meetings[meeting] = imp_rates
    meetings_implied_df = pd.concat(meetings.values(),axis = 1).fillna(0)
    
    meetings_implied_df = meetings_implied_df.round().astype(int) #round up/down decimals 
    meetings_implied_df.replace(0, np.nan, inplace=True)  #convert all 0s to nans (contract isn't open yet)
    
    return meetings_implied_df


# ## Interactive plotting

# In[12]:


def plotting(meetings_implied_df,date):
    '''
    Takes in a meetings_implied_df and datelist(date slider), plots 2 yield curves
    Also returns data table of all implied rates for different meetings on the two input days     
    '''
    
    meetings = np.array(meetings_implied_df.columns)
    less_than_two_contracts_2 = False
    less_than_two_contracts_1 = False

    
    date_2 = date[-1]
    date_1 = date[0]
    unfiltered_row_2 = np.array(meetings_implied_df.loc[date_2])
    unfiltered_row_1 = np.array(meetings_implied_df.loc[date_1])
    
    row_2 = unfiltered_row_2[~np.isnan(unfiltered_row_2)]   # some days not all contracts are active, this is required so that our smoothing can work properly
    row_1 = unfiltered_row_1[~np.isnan(unfiltered_row_1)]   # some days not all contracts are active
   
    
    str_to_dt = np.vectorize(lambda x: datetime.strptime(x, '%y%b'))
    meeting_dates = str_to_dt(meetings) #all meeting dates 

    
    meeting_timedelta_from_start = (meeting_dates-meeting_dates[0])
    dt_to_days = np.vectorize(lambda x: x.days)
    meeting_days_from_start = dt_to_days(meeting_timedelta_from_start)  #all meeting days from start
    
    row2_meeting_days_from_start = meeting_days_from_start[~np.isnan(unfiltered_row_2)]
    row1_meeting_days_from_start = meeting_days_from_start[~np.isnan(unfiltered_row_1)]
    
    day_first = 0
    day_last_2 = row2_meeting_days_from_start[-1]
    day_last_1 = row1_meeting_days_from_start[-1]

    #applying curve smoothing
    try: 
        smoothed_2 = CubicSpline(row2_meeting_days_from_start,row_2)
        fitted_range_2 = np.linspace(day_first,day_last_2)
    except:     #<2 conrtacts available on that day
        less_than_two_contracts_2 = True
    try:
        smoothed_1 = CubicSpline(row1_meeting_days_from_start, row_1)
        fitted_range_1 = np.linspace(day_first,day_last_1)
    except:
        less_than_two_contracts_1 = True

    #plotting
    plt.figure(figsize=(10, 4))
    
    #plot rates
    plt.scatter(x = meeting_days_from_start, y =unfiltered_row_2,label = f'Implied curve on {date_2.date()}')    
    plt.scatter(x = meeting_days_from_start, y =unfiltered_row_1,label = f'Implied curve on {date_1.date()}') 
    #plot interpolated curve 
    if less_than_two_contracts_2 == False:
        plt.plot(fitted_range_2, smoothed_2(fitted_range_2))
    if less_than_two_contracts_1 == False:
        plt.plot(fitted_range_1, smoothed_1(fitted_range_1))
    xticks(meeting_days_from_start, meetings)  # Space out meetings according to time
    
    plt.title('Historical Implied Fed Funds Rate/bps Across Meetings')
    plt.legend(loc="upper left")
    plt.show()
    
    #returns data table of all implied rates for different meetings on the two input days 
    table = pd.DataFrame(np.array([unfiltered_row_1,unfiltered_row_2]),columns = meetings, index = [date_1,date_2])
    
    return table

def plot_historical_curve(meetings_implied_df):
    '''
    Turn the plotting function from static to interactive
    '''
    
    #Building our Date Slider
    start_date = meetings_implied_df.index[0]
    end_date = meetings_implied_df.index[-1]
    dates = pd.date_range(start_date, end_date, freq='D')
    options = [(date.strftime(' %d %b %y '), date) for date in dates]
    index = (0, len(options)-1)

    selection_range_slider = widgets.SelectionRangeSlider(
        options=options,
        index=index,
        description='Dates',
        orientation='horizontal',
        layout={'width': '600px'}
    )

    return widgets.interact(plotting,meetings_implied_df = widgets.fixed(meetings_implied_df),date = selection_range_slider)


# ## Overall construction

# ### Inputting some default data

# In[ ]:


def default_meetings():
    '''
    Default meetings are the most recently closed meeting + all future unclosed meetings
    '''
    actual_FOMC = User.actual_FOMC
    most_recent_id_FED = 'FED-'+get_recent_close(User.actual_FOMC)[0]
    defaults = [most_recent_id_FED] + unsettled_contracts()
    return defaults


# ## Exploring data anomolies

# In[ ]:


def check_meeting(meeting,lookback):
    '''
    meeting:a string of the form '23FEB'; lookback: int, # of lookback days(should be same as the one used in plotting)
    returns 1 df and 1 dictionary
    df1 shows historical prices of all contracts; 
    dic1 contains more detailed historical data for different specific contracts (output of function build_df_each_rate);
    Each df in dic1 has index Time, columns [volume, open_interest, price, spread] 
    '''
    series_id = 'FED-'+meeting   #convert to ideal format
   
    #get df for the target series_id in order to retrieve all tickers for that meeting
    series_content_df = specified_markets(series_ticker= series_id)
    #Get dictionary of df for specific strikes
    try:
        contracts = build_df_each_rate(contract_ids = series_content_df.ticker.tolist(), categories = series_content_df.subtitle.tolist(),lookback = lookback) 
    except:
        raise ValueError(f'Please check your lookback period or meetings are correct!')
    
    meetings = {}
    cum_df = cumulative_df(contracts).iloc[:,::-1]
    imp_rates = discrete_df(cum_df,month = series_id[-3:], year =series_id[-5:-3]) #naming the columns
    meetings[series_id] = imp_rates
    meetings_implied_df_ = pd.concat(meetings.values(),axis = 1).fillna(0)
    
    cum_df.reset_index(drop = False,inplace = True)
    cum_df['Date']=cum_df['Date'].apply(lambda x: x.date())
    return cum_df, contracts

def check_meeting_ondate(cum_df,year,month,day):
    '''
    e.g. year = 2022,month = 12,day = 20
    returns closing mid prices of all contracts on that date
    '''
    date_to_check = datetime(year,month,day).date()
    return cum_df[cum_df['Date']==date_to_check]

def check_raw_data(contracts,strike,year,month,day):
    '''
    contracts: our dictionary of dfs for different strikes for that meeting
    strike: string, '2.00%', obtained by inspecting columns from output of check_meeting_ondate()
    returns data on on specific strike contract for our given meeting on our specific date 
    '''
    all_contracts = list(contracts.keys())
    
    #filter for the index that contains the strike  
    #our strike may be'2.00%', yet our contract_index may be '>2.00%'
    specific_contract = [x for x in all_contracts if strike in x][0]
    raw_data = contracts[specific_contract]
    raw_data['Date']=raw_data['Date'].apply(lambda x: x.date())
    date_to_check = datetime(year,month,day).date()
    row = raw_data[raw_data['Date']==date_to_check]
    
    return row


# In[ ]:


# historical_prices,contracts = check_meeting('23JUN',lookback = 30)


# In[ ]:


# check_meeting_ondate(historical_prices,2022,12,2)


# In[ ]:


# check_raw_data(contracts, '5.00%').head()


# ## Code below for testing output & debugging

# #### 1.Testing various intermediate outputs

# In[ ]:


# series_id='FED-23JUN'
# lookback_ = 10
# series_dfs = {}

# #get df for each contract in the series in order to retrieve all tickers for that series
# series_content_df = specified_markets(series_ticker= series_id)
# #for each series, get dictionary of df for specific contracts
# contracts = build_df_each_rate(contract_ids = series_content_df.ticker.tolist(), categories = series_content_df.subtitle.tolist(),lookback = lookback_) 
# #we put the dictionary into another dictionary
# series_dfs[series_id] = contracts
    
# meetings = {}
# cum_df = cumulative_df(contracts).iloc[:,::-1]
# imp_rates = discrete_df(cum_df,month = series_id[-3:], year =series_id[-5:-3]) #naming the columns
# meetings[series_id] = imp_rates
# meetings_implied_df_ = pd.concat(meetings.values(),axis = 1).fillna(0)


# In[ ]:


# cum_df.reset_index(drop = False,inplace = True)
# cum_df['Date']=cum_df['Date'].apply(lambda x: x.date())
# cum_df.head()


# In[ ]:


# date_to_check = datetime(2022,12,20).date()
# cum_df[cum_df['Date']==date_to_check]


# In[ ]:


# series_dfs['FED-23FEB']['>2.00%']


# #### 2.Testing plotting

# In[13]:


# meetings_ = ['FED-23SEP','FED-23NOV']
# lookback_ = 10


# In[14]:


# meetings_implied_df = get_meetings_implied_df(series_ids = meetings_,lookback = lookback_)


# In[17]:


# #Plotting dashboard
# plot_historical_curve(meetings_implied_df)

