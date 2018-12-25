import fix_yahoo_finance as yf

# Use this ftp as an initial test to retrieve data
# Not sure how to get dutch market
# ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqlisted.txt

ticker_list = ['TSLA', 'GOOGL']
finance_data = yf.download(ticker_list[0])

import matplotlib.pyplot as plt
z = finance_data['Close']
plt.plot(z)