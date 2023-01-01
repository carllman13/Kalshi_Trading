# Kalshi_Trading
All Kalshi-related projects - data analysis/visualization/trading

FedCurve: 
A package for analyzing implied Fed Funds Rates Curve over time.

What it can do:
Interactive charting/data table for implied Fed Funds Rate on different meeting dates in past X days. 
Today, I can use prices of different Fed-Rate related contracts on Kalshi to calculated implied Fed Funds rate for different meetings.
For example, if I have rates for 23FEB, 23MAR,23MAY (3 meetings), I can plot this and this becomes a curve.
Everyday since the price of the contracts for each meeting changes, my curve changes. 
This package allows you to visualize how the curve has changed over time and allows you to easily compare curves.

Instructions:
There are 3 files, Main_script, FedCurve, User. Each has an .ipynb and .py version. I prefer to run the ipynb and save changes to py. 
FedCurve contains all functions.
Main_script is where you will call the functions (some example parameters have been written)
User is for setting username/pw/other default parameters.

