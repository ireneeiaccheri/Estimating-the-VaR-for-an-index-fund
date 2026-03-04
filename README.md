This quantitative finance project demonstrates how to best describe and quantify risk in time series data through an analysis performed in MATLAB. It compares different approaches, with a primary focus on the GARCH model.

The main idea is to estimate Value at Risk (VaR) and see how traditional models hold up against the GARCH approach. Real-world financial markets frequently experience "volatility clustering," and by comparing the computations and plots—as detailed in the accompanying paper—we concluded that the GARCH model more accurately captures potential losses.

The original analysis was performed on the Standard & Poor’s Europe 350 index, using daily closing prices from April 2014 to April 2019. However, the code is designed to be highly adaptable. You can easily swap out our dataset for one of your own, allowing you to run the same analysis and comparison on any time series to draw your own conclusions.
