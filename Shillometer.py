#Need to pip install seaborn, ccxt, matplotlib, scikit-learn on each restart

#conda activate /Users/stevegoulden/opt/anaconda3/envs/myenv

import streamlit as st
#from functions import merge_coin_prices, coin_df, shill_chart, group_data

#pull_prices

import pandas as pd
import re
import requests
from datetime import datetime, timedelta, date
import numpy as np
import pytz
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
import matplotlib.dates as mdates

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import TimeSeriesSplit
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import statistics

import ccxt

import requests
import csv
import time


#Telegram and huggingface done in colab and then most recent file saved
# This uses the file Shillometer_TG_HF.ipynb

st.set_page_config(layout="wide")
st.title('Welcome to Shillometer')

st.markdown("""
## We scrape high volume Telegram crypto groups, which we then run through a Hugging Face sentiment library

### From that we calculate 'Shill score' = average sentiment * number of mentions

### We combine this with technical indicators and use k-means clustering and PCA to identify the high conviction entry points

### First off, here are the best and worst shills today and this week

""")




def pull_prices(days, coin):
    exchange_id = 'binance'
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({
        'timeout': 30000,
        'enableRateLimit': True,
    })

    symbol = coin
    symbol = f'{coin}/USDT'
    timeframe = '1d'

    seven_days_ago = datetime.utcnow() - timedelta(days=days)
    since = int(seven_days_ago.timestamp() * 1000)  # Convert to milliseconds

    all_ohlcv = []
    limit = 500  # Adjust based on exchange limits

    # Fetch OHLCV data starting from x days ago, handling pagination if necessary
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        if len(ohlcv) == 0:
            break  # No more data to fetch
        all_ohlcv += ohlcv
        since = ohlcv[-1][0] + 1  # Update 'since' to get the next batch of data

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df_prices = df[['timestamp', 'close', 'volume']]
    df_prices = df_prices.rename(columns={'timestamp': 'Date', 'close': 'price', 'volume':'24h_volume'})

    return df_prices

#Coingecko pull prices for the on chain
def pull_prices_cg(days, coin):
    
    coin = coin.lower()

    filtered_series = coins_list[coins_list['Ticker'] == coin]['ID']

    # To get the index of the coin in the filtered series, wihc will be used in a loc lookuip
    indices = filtered_series.index

    if not indices.empty:
        first_index = indices[0]  # Gets the first index of the filtered series
        coin_id = coins_list.loc[first_index, 'ID']
    
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    parameters = {
        'vs_currency': 'usd',
        'days': days,
    }
    response = requests.get(url, params=parameters)
    data = response.json()

    df_prices = pd.DataFrame(data['prices'], columns=['time', 'price'])
    df_volumes = pd.DataFrame(data['total_volumes'], columns=['time', '24h_volume'])
    
    
    df_prices['Date'] = pd.to_datetime(df_prices['time'], unit='ms').dt.date  # Converts timestamp to date
    sorted_prices = df_prices.sort_values(by='time')
    daily_closing_prices = sorted_prices.drop_duplicates('Date', keep='last')[['Date', 'price']]
    daily_closing_prices.reset_index(inplace=True, drop=True)
    daily_closing_prices['Date'] = pd.to_datetime(daily_closing_prices['Date'])

    df_volumes['Date'] = pd.to_datetime(df_volumes['time'], unit='ms').dt.date  # Converts timestamp to date
    sorted_volumes = df_volumes.sort_values(by='time')
    daily_closing_volumes = sorted_volumes.drop_duplicates('Date', keep='last')[['Date', '24h_volume']]
    daily_closing_volumes.reset_index(inplace=True, drop=True)
    daily_closing_volumes['Date'] = pd.to_datetime(daily_closing_volumes['Date'])
    
    df_price = pd.merge(daily_closing_prices, daily_closing_volumes, on='Date')
    
    
    return df_price

def fetch_cryptocurrencies(per_page=250):
    url = "https://api.coingecko.com/api/v3/coins/markets"
    page = 1
    all_cryptos = []

    while True:
        params = {
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'per_page': per_page,
            'page': page
        }
        response = requests.get(url, params=params)
        if response.status_code != 200:
            break
        data = response.json()
        if not data:
            break
        all_cryptos.extend(data)
        page += 1

    return all_cryptos

def save_to_csv_using_dataframe(cryptos, filename):
    df = pd.DataFrame(cryptos)

    df = df[['id', 'name', 'symbol', 'market_cap', 'total_volume']]
    df.columns = ['ID', 'Name', 'Ticker', 'Market Cap', '24h Volume']

    df.to_csv(filename, index=False)


def merge_coin_prices(df, master, coin):
    new_df = pd.concat([df, master], axis = 0)# sort=True)
    new_df = new_df.drop_duplicates(subset=['Date'])
    new_df['Date'] = pd.to_datetime(new_df['Date'])
    new_df = new_df.sort_values(by='Date')
    master = new_df
    master.to_csv(f'coin_prices/{coin}.csv')

    return new_df



#NB deleted original, this breaks all coin refs out from original list

def group_data(df):
    
    # Convert 'coin_mentions' from string representation of list to actual list
    df['date'] = pd.to_datetime(df[' message.date']).dt.date
    
    #df['coin_mentions'] = df['coin_mentions'].apply(eval)
    
    #This creates a df with each comment having its own line, with mentions and avg sentiment
    #explode - each coin mention get own line
    
    exploded_telegram_comments_df = df.explode('coin_mentions')

    #groupby date and then coin mentions and add num_mentions as count of message.text and 
    #avg_sentiment as mean sentiment score. Then reset index
    grouped_telegram_comments_data = exploded_telegram_comments_df.groupby(['date', 'coin_mentions']).agg(
        num_mentions=pd.NamedAgg(column="message.text", aggfunc="count"),
        avg_sentiment_score=pd.NamedAgg(column="sentiment_score", aggfunc="mean")
    ).reset_index()


    # Filter out rows with no coin mentions
    grouped_telegram_comments_data = grouped_telegram_comments_data[grouped_telegram_comments_data['coin_mentions'].notna()]

    grouped_telegram_comments_data['date'] = pd.to_datetime(grouped_telegram_comments_data['date'])
    
    return grouped_telegram_comments_data



def label_return_final(df):

    df['log_price'] = np.log(df['price'])
    df['log_rtn'] = df['log_price'].diff()

    
    summary_df = pd.DataFrame(columns=['Cluster', 'Count', 'Hit Rate', 'Mean', 'Std', 'Min', '50%', 'Max'])

    # Parameters for the strategy
    holding_period = 14
    stop_loss = 0.10  # 10%
    
    fig, ax1 = plt.subplots(figsize=(14, 8))  # Adjust the figure size as needed
    #colors = plt.cm.tab10(np.linspace(0, 1, len(df['labels'].unique())))
    colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'orange', 4: 'cyan'}


    for label in df['labels'].unique():
        df['trigger'] = np.where(df['labels'] == label, 1, 0)
        df['stance'] = 0
        df['in_trade'] = False
        df['trade_entry_price'] = None
        df['in_trade_rtn'] = 0
        df['14d_rtn'] = 0
        df['14d_fwd_rtn'] = 0
        
        for i in range(len(df) - holding_period -1):
            if df.loc[i, 'trigger'] == 1 and not df.loc[i, 'in_trade']:
                df.loc[i+1, 'in_trade'] = True
                df.loc[i + 1, 'stance'] = 1
                df.loc[i, 'trade_entry_price'] = df.loc[i, 'price']
                
                df.loc[i+holding_period, '14d_fwd_rtn'] = (df.loc[i+holding_period, 'price'] / df.loc[i, 'price'])-1
                
                for j in range(i + 1, i + holding_period + 1):
                    df.loc[j, 'stance'] = 1
                    df.loc[j, 'in_trade'] = True
                    df.loc[j, 'trade_entry_price'] = df.loc[i, 'trade_entry_price']

                    df.loc[j, 'in_trade_rtn'] = (df.loc[j, 'price'] / df.loc[i, 'trade_entry_price']) - 1
                    if df.loc[j, 'in_trade_rtn'] < -stop_loss or (j - i) == holding_period:
                        df.loc[j + 1, 'trigger'] = -1
                        df.loc[j + 1, 'stance'] = 0
                        df.loc[j + 1, 'in_trade'] = False
                   
                        df.loc[i, 'trade_rtn'] = df.loc[j, 'price'] / df.loc[i, 'price']
                        break  # Exit the trade loop upon stop loss trigger or end of holding period
                        
            if df.loc[i, 'trigger'] == 1 and df.loc[i, 'in_trade']:
                df.loc[i+1, 'in_trade'] = True
                df.loc[i + 1, 'stance'] = 1
                df.loc[i, 'trade_entry_price'] = df.loc[i-1, 'trade_entry_price']
                df.loc[i+holding_period, '14d_fwd_rtn'] = (df.loc[i+holding_period, 'price'] / df.loc[i, 'price'])-1
                
                for j in range(i + 1, i + holding_period + 1):
                    df.loc[j, 'stance'] = 1
                    df.loc[j, 'in_trade'] = True
                    df.loc[j, 'trade_entry_price'] = df.loc[i, 'trade_entry_price']
                    
                    df.loc[j, 'in_trade_rtn'] = (df.loc[j, 'price'] / df.loc[i, 'trade_entry_price']) - 1
                    if df.loc[j, 'in_trade_rtn'] < -stop_loss or (j - i) == holding_period:
                        df.loc[j + 1, 'trigger'] = -1
                        df.loc[j + 1, 'stance'] = 0
                        df.loc[j + 1, 'in_trade'] = False
                        
                        df.loc[i, 'trade_rtn'] = df.loc[j, 'price'] / df.loc[i, 'price']
                        
                        #df.loc[i, 'trade_rtn'] = df.loc[j, 'in_trade_rtn']
                        break  # Exit the trade loop upon stop loss trigger or end of holding period

                    

                
        # Calculate equity log return
        df['equity_log_return'] = np.where(df['stance'] == 1, df['log_rtn'], 0)
        df['cumulative_equity_return'] = df['equity_log_return'].cumsum().apply(np.exp)
        




        # Adjust the initialization of the equity curve to reflect initial investment accurately
        df['equity_curve'] = df['cumulative_equity_return']

        df['date'] = pd.to_datetime(df['date'])
        
        #return df
        #df.to_csv(f'df_{label}_k_means_250224.csv')

 #       return df[['date', 'price', 'trigger', 'stance', 'in_trade', 'trade_entry_price', 'trade_rtn', 'labels']].head(50)


        #calc hit rate
        hit_rate = len(df[df['14d_fwd_rtn'] >0]['14d_rtn'])/len(df[df['14d_fwd_rtn'] !=0]['14d_fwd_rtn'])

        stats = df[df['14d_fwd_rtn'] !=0]['14d_fwd_rtn'].describe()

        summary_df.loc[len(summary_df)] = [
            label,
            stats['count'],
            hit_rate,
            stats['mean'],
            stats['std'],
            stats['min'],
            stats['50%'],
            stats['max']]


        ax1.plot(df['date'], df['cumulative_equity_return'], label=f'Label {label}', color=colors[label])   

        
       
        #df.to_csv(f'df_{label}_k_means_250224.csv')

        ax1.set_xlabel('Date')
        ax1.set_ylabel('Cumulative Equity Return')
        ax1.tick_params(axis='y')
        ax1.legend(loc='upper left')


    normalized_price = df['price'] / df['price'].iloc[0]

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Normalized Price', color='black')  # we already handled the x-label with ax1
    ax2.plot(df['date'], normalized_price, color='cyan')
    ax2.tick_params(axis='y', labelcolor='black')

    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # Set the x-ticks rotation
    for label in ax1.get_xticklabels():
        label.set_rotation(45)
    #plt.xticks(rotation=45)

    plt.tight_layout() 
    plt.title('Cumulative Equity Return and Normalized Price Over Time')
    plt.show()    
    st.pyplot(fig)
    summary_sorted = summary_df.sort_values(by='Hit Rate', ascending=False)
    return summary_sorted
    


def buy_signals_over_time(df, k_means_df):
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)

    fig = plt.figure(figsize=(14, 7))

    # Plot price data on primary y-axis
    ax1 = plt.gca()  # Get current axis
    ax1.plot(df['date'], df['price'], label='Price', color='cyan', linewidth=1)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price', color='gray')
    ax1.tick_params(axis='y', labelcolor='gray')

    # Create a twin y-axis for the labels
    ax2 = ax1.twinx()
    ax2.set_ylabel('Label', color='black')  # Set label for the right y-axis

    # Define colors for each label
    colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'orange', 4: 'cyan'}


    # Plot buy signals on the twin y-axis
    for label, group in k_means_df.groupby('labels'):
        ax2.scatter(group['date'], [label]*len(group), color=colors[label], label=f'Label {label}', alpha=0.6, edgecolors='w')

    # Since we're using a secondary axis, the legend will only show entries for one of the axes
    # To include both, manually combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.title('Buy Signals Over Time')
    plt.tight_layout()
    plt.show()
    st.pyplot(fig)

def kmeans_pca(df):
    features = ['num_mentions', 'avg_sentiment_score', 'shill_score', '10_20_ema', '10_20_acc', '14d_RSI', '3d_change_CVD', 'AO', 'AC_CVD',
           'shill_score_7d_ema', 'avg_sentiment_score_7d_ema', 'num_mentions_7d_ema', 'BB_Width', 'P_vs_lower']

    #df_pca = df[:-14][features]
    df_pca = df[features]
    df_pca=df_pca.fillna(0)
    scaler = StandardScaler()
    segmentation_std = scaler.fit_transform(df_pca)

    from sklearn.decomposition import PCA
    pca = PCA()
    #pca.fit(segmentation_std)
    #pca.explained_variance_ratio_
    pca = PCA(n_components = 4, random_state=42)

    pca.fit(segmentation_std)
    pca.transform(segmentation_std)
    scores_pca = pca.transform(segmentation_std)

    kmeans_pca = KMeans(n_clusters = 5, random_state=42)
    kmeans_pca.fit(scores_pca)

    df_pca_kmeans = pd.concat([df_pca.reset_index(drop=True), pd.DataFrame(scores_pca)], axis = 1)
    df_pca_kmeans.columns.values[-4:] = ['PC1', 'PC2', 'PC3', 'PC4']
    df_pca_kmeans['Segment K-means PCA'] = kmeans_pca.labels_
    df_pca_kmeans['Segment'] = df_pca_kmeans['Segment K-means PCA'].map({0: 'first', 1: 'second', 2: 'third', 3: 'fourth', 4: 'fifth'})

    # Assuming 'df_ZZ_pca_kmeans' is your DataFrame and it has 'PC3' for the third principal component.
    x_axis = df_pca_kmeans['PC1']
    y_axis = df_pca_kmeans['PC2']
    z_axis = df_pca_kmeans['PC3']

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot adding a loop for color because seaborn's scatterplot doesn't work in 3D.
    colors = sns.color_palette('bright', n_colors=len(df_pca_kmeans['Segment'].unique()))
    for segment, color in zip(df_pca_kmeans['Segment'].unique(), colors):
        ix = df_pca_kmeans['Segment'] == segment
        ax.scatter(x_axis[ix], y_axis[ix], z_axis[ix], c=[color], label=segment, s=50)



    df_pca_kmeans = df_pca_kmeans.rename(columns = {'Segment K-means PCA': 'labels'} )
    #Adding new lines

#cutting these as they mess up the charts

    df_pca_kmeans['14d_rtn'] = df['14d_rtn']
    df_pca_kmeans['14d_fwd_rtn'] = df['14d_rtn']
    df_pca_kmeans['14d_rtn'] = df_pca_kmeans['14d_rtn'].fillna(0)
    df_pca_kmeans['14d_fwd_rtn'] = df_pca_kmeans['14d_fwd_rtn'].fillna(0)

#    df_pca_kmeans['date'] = df['date'][:-14]
#    df_pca_kmeans['price'] = df['price'][:-14]
    df_pca_kmeans['date'] = df['date']
    df_pca_kmeans['price'] = df['price']

    return df_pca_kmeans

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.title('Clusters by PCA Components in 3D')
    plt.legend()
    plt.show()
    st.pyplot(fig)



#NB this is meant to ingest df_pca_kmeans
def dynamic_label_return_strategy(df):
    tscv = TimeSeriesSplit(n_splits=20)
    overall_returns = []
    equity_curves = []
    top_clusters = []
    test_data_parts = []
    trade_dates = []    
    
    features = ['PC1', 'PC2', 'PC3', 'PC4']
    features_price = ['date', 'PC1', 'PC2', 'PC3', 'PC4', 'price', '14d_fwd_rtn']
    df = df[features_price]
    
    df['log_price'] = np.log(df['price'])
    df['log_rtn'] = df['log_price'].diff()
    df.dropna(inplace=True)
    

    
    summary_df = pd.DataFrame(columns=['Cluster', 'Count', 'Hit Rate', 'Mean', 'Std', 'Min', '50%', 'Max'])

    # Parameters for the strategy
    holding_period = 14
    stop_loss = 0.10  # 10%
    
    
    for fold, (train_index, test_index) in enumerate(tscv.split(df)):
        train_data, test_data = df.iloc[train_index].copy(), df.iloc[test_index].copy()
        kmeans = KMeans(n_clusters=5, random_state=fold).fit(train_data[['PC1', 'PC2', 'PC3', 'PC4']])
        train_labels = kmeans.predict(train_data[['PC1', 'PC2', 'PC3', 'PC4']])
        test_labels = kmeans.predict(test_data[['PC1', 'PC2', 'PC3', 'PC4']])
        test_data['predicted_label'] = test_labels
        test_data_parts.append(test_data)
        

        # Determine the top-performing cluster
        train_data['label'] = train_labels
        top_cluster = train_data.groupby('label')['14d_fwd_rtn'].mean().idxmax()
        top_clusters.append(top_cluster)

        test_data['is_top_cluster'] = test_data['predicted_label'] == top_cluster

        # Initialize columns for trading logic
        test_data['in_trade'] = False
        test_data['trade_entry_price'] = np.nan
        test_data['exit_price'] = np.nan
        test_data['trade_return'] = np.nan
        test_data['entry'] = np.nan
        

        in_trade = False
        entry_index = None

        
        
        #This is where the new strategy comes in
        for i in range(len(test_data)):
            if test_data.iloc[i]['is_top_cluster'] and not in_trade:
                in_trade = True
                #test_data['entry'] == 1
                test_data.iloc[i, test_data.columns.get_loc('entry')] = 1
                entry_index = i
                entry_date = test_data.iloc[i]['date']
                trade_dates.append(entry_date)

            if in_trade and i >= entry_index:
                if entry_index == i:  # Set entry price at open of trade
                    test_data.iloc[i, test_data.columns.get_loc('trade_entry_price')] = test_data.iloc[i]['price']
                    entry_price = test_data.iloc[i]['price']

                current_price = test_data.iloc[i]['price']
                current_return = (current_price - entry_price) / entry_price

                # Check for stop loss or holding period end
                if current_return < -stop_loss or (i - entry_index) >= holding_period:
                    test_data.iloc[i, test_data.columns.get_loc('exit_price')] = current_price
                    test_data.iloc[i, test_data.columns.get_loc('trade_return')] = current_return
                    overall_returns.append(current_return)
                    in_trade = False  # Reset trade status

        # Process final equity and returns
        if in_trade:  # Handle open trade at the end of the test data
            current_price = test_data.iloc[-1]['price']
            current_return = (current_price - entry_price) / entry_price
            test_data.iloc[-1, test_data.columns.get_loc('exit_price')] = current_price
            test_data.iloc[-1, test_data.columns.get_loc('trade_return')] = current_return
            overall_returns.append(current_return)
        
        #test_data.to_csv('test_data_sol.csv')
    
    
    full_test_data = pd.concat(test_data_parts)
    full_test_data.reset_index(drop=True, inplace=True)
    
    #full_test_data.to_csv('full_test_data_sol_1.csv')
    
    
    #Now building strategy
    
    
    full_test_data['trigger'] = np.where(full_test_data['is_top_cluster'] == True, 1, 0)
    full_test_data['stance'] = 0
    full_test_data['in_trade'] = False
    full_test_data['trade_entry_price'] = None
    full_test_data['in_trade_rtn'] = 0
    full_test_data['14d_rtn'] = 0
    full_test_data['14d_fwd'] = 0
    
    #full_test_data.to_csv('full_test_data_sol_0.csv')

    for i in range(len(full_test_data) - holding_period -1):
        if full_test_data.loc[i, 'trigger'] == 1 and not full_test_data.loc[i, 'in_trade']:
            full_test_data.loc[i+1, 'in_trade'] = True
            full_test_data.loc[i + 1, 'stance'] = 1
            full_test_data.loc[i, 'trade_entry_price'] = full_test_data.loc[i, 'price']

            full_test_data.loc[i+holding_period, '14d_fwd'] = (full_test_data.loc[i+holding_period, 'price'] / full_test_data.loc[i, 'price'])-1

            for j in range(i + 1, i + holding_period + 1):
                full_test_data.loc[j, 'stance'] = 1
                full_test_data.loc[j, 'in_trade'] = True
                full_test_data.loc[j, 'trade_entry_price'] = full_test_data.loc[i, 'trade_entry_price']

                full_test_data.loc[j, 'in_trade_rtn'] = (full_test_data.loc[j, 'price'] / full_test_data.loc[i, 'trade_entry_price']) - 1
                if full_test_data.loc[j, 'in_trade_rtn'] < -stop_loss or (j - i) == holding_period:
                    full_test_data.loc[j + 1, 'trigger'] = -1
                    full_test_data.loc[j + 1, 'stance'] = 0
                    full_test_data.loc[j + 1, 'in_trade'] = False

                    full_test_data.loc[i, 'trade_rtn'] = full_test_data.loc[j, 'price'] / full_test_data.loc[i, 'price']
                    break  # Exit the trade loop upon stop loss trigger or end of holding period

    if full_test_data.loc[i, 'trigger'] == 1 and full_test_data.loc[i, 'in_trade']:
        full_test_data.loc[i+1, 'in_trade'] = True
        full_test_data.loc[i + 1, 'stance'] = 1
        full_test_data.loc[i, 'trade_entry_price'] = full_test_data.loc[i-1, 'trade_entry_price']
        full_test_data.loc[i+holding_period, '14d_fwd_rtn'] = (full_test_data.loc[i+holding_period, 'price'] / full_test_data.loc[i, 'price'])-1

        for j in range(i + 1, i + holding_period + 1):
            full_test_data.loc[j, 'stance'] = 1
            full_test_data.loc[j, 'in_trade'] = True
            full_test_data.loc[j, 'trade_entry_price'] = full_test_data.loc[i, 'trade_entry_price']

            full_test_data.loc[j, 'in_trade_rtn'] = (full_test_data.loc[j, 'price'] / full_test_data.loc[i, 'trade_entry_price']) - 1
            if full_test_data.loc[j, 'in_trade_rtn'] < -stop_loss or (j - i) == holding_period:
                full_test_data.loc[j + 1, 'trigger'] = -1
                full_test_data.loc[j + 1, 'stance'] = 0
                full_test_data.loc[j + 1, 'in_trade'] = False

                full_test_data.loc[i, 'trade_rtn'] = full_test_data.loc[j, 'price'] / full_test_data.loc[i, 'price']

                #full_test_data.loc[i, 'trade_rtn'] = full_test_data.loc[j, 'in_trade_rtn']
                break  # Exit the trade loop upon stop loss trigger or end of holding period




    # Calculate equity log return
    full_test_data['equity_log_return'] = np.where(full_test_data['stance'] == 1, full_test_data['log_rtn'], 0)
    full_test_data['cumulative_equity_return'] = full_test_data['equity_log_return'].cumsum().apply(np.exp)
    
    return full_test_data
    #full_test_data.to_csv('full_test_data_sol_2.csv')

#use full_test_data on the below
#def return_hit_rate(df):

    
    

    # Calculate cumulative returns
    #final_return = np.prod([1 + rtn for rtn in overall_returns if not np.isnan(rtn)]) - 1
    #final_return = df['cumulative_equity_return'].iloc[-1]
    
#    hit_rate = len(list(filter(lambda overall_returns: overall_returns > 0, overall_returns))) / len(overall_returns)

 #   average_return = statistics.mean(overall_returns) if overall_returns else 0

  #  return final_return, average_return, hit_rate
    
    #print(f"Final cumulative return: {final_return * 100:.2f}%", f"Average return: {statistics.mean(overall_returns)}", f"Hit Rate: {hit_rate}", f"Clusters: {top_clusters}")
    #print(overall_returns)
    
    
#    results_df = pd.DataFrame({
#    'Trade Date': trade_dates,
#    'Return': overall_returns
#})
#    summary_stats = pd.DataFrame({
#    'Metric': ['Final Cumulative Return', 'Average Return', 'Hit Rate'],
#    'Value': [f"{final_return * 100:.2f}%", f"{average_return}", f"{hit_rate}"]
#})

def dynamic_label_return_strategy_plot(df):
    # Plot the equity curve using 'date' for x-axis and 'cumulative_equity_return' for y-axis
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 8))
    plt.plot(df['date'], df['cumulative_equity_return'], label='Equity Curve', color='#77dd77')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Equity Return')
    plt.title('Equity Curve Over Time')
    plt.legend()
    #plt.show()
    st.pyplot(fig)

    
    
    #return results_df, summary_stats

#Amend the below for k means not rf


#


def coin_df(days, coin, df):
    df['date'] = pd.to_datetime(df[' message.date']).dt.date

    grouped_telegram_comments_data = group_data(df)

    coin_data = grouped_telegram_comments_data[grouped_telegram_comments_data['coin_mentions'].apply(lambda x: coin.upper() in x)]

    #now trying to see rolling dates, with some entries empty so can compare to eg price data
    #build new df with date starting 14 July, ending on last post
    all_dates = pd.DataFrame(pd.date_range(start='2020-07-01', end=coin_data['date'].max()), columns=['date'])

    #merge comments and price data
    merged_coin_data = pd.merge(all_dates, coin_data, how='left', on='date')

    # create shill score
    merged_coin_data['shill_score'] = merged_coin_data['num_mentions'] * merged_coin_data['avg_sentiment_score']

    # fillna
    merged_coin_data['num_mentions'] = merged_coin_data['num_mentions'].fillna(0)
    merged_coin_data['shill_score'] = merged_coin_data['shill_score'].fillna(0)
    merged_coin_data['avg_sentiment_score'] = merged_coin_data['avg_sentiment_score'].fillna(0)
     
    df_price = pull_prices(days, coin)
     
    merged_coin_prices = df_price   
    
    
    start_date = datetime.strptime('2020-01-01', '%Y-%m-%d')

    # Filter the dataframes
    filtered_coin_data = merged_coin_data[merged_coin_data['date'] >= start_date]
    filtered_price_data = merged_coin_prices[merged_coin_prices['Date'] >= start_date]
    
    #rename date col
    filtered_price_data = filtered_price_data.rename({'Date':'date'}, axis=1)

    #merge
    df = pd.merge(filtered_price_data, filtered_coin_data, how='left', on='date')
    
    
    # Technicals
    df['20d_ema'] = df['price'].ewm(span=20, adjust=False).mean()
    df['10d_ema'] = df['price'].ewm(span=10, adjust=False).mean()
    df['5d_ema'] = df['price'].ewm(span=5, adjust=False).mean()
    
    
    df['10_20_ema'] = df['10d_ema'] / df['20d_ema'] 
    df['5_10_ema'] = df['5d_ema'] / df['10d_ema'] 
    
    df['10_20_acc'] = df['10_20_ema'] / df['10_20_ema'].rolling(window=5).mean()
    df['5_10_acc'] = df['5_10_ema'] / df['5_10_ema'].rolling(window=5).mean()

    # 14 day RSI
    delta = df['price'].diff(1)
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['14d_RSI'] = 100 - (100 / (1 + rs))
    
    # 7 day RSI 
    delta = df['price'].diff(1)
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/7, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/7, adjust=False).mean()
    rs = gain / loss
    df['7d_RSI'] = 100 - (100 / (1 + rs))

    # Calculate CVD
    df['price_change'] = df['price'].diff()
    df['volume_direction'] = df['24h_volume'] * df['price_change'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    df['CVD'] = df['volume_direction'].cumsum()

    # Drop columns
    df.drop(columns=['price_change', 'volume_direction'], inplace=True)
    
    # Z-Score of 3-Day Change in CVD
    df['3d_change_CVD'] = df['CVD'].diff(3)
    rolling_mean = df['3d_change_CVD'].rolling(window=30).mean()
    rolling_std = df['3d_change_CVD'].rolling(window=30).std()
    df['z_score_3d_change_CVD'] = (df['3d_change_CVD'] - rolling_mean) / rolling_std
    
    # AC of CVD - similar concept to z-score
    df['5d_sma_CVD'] = df['CVD'].rolling(window=5).mean()
    df['AC_CVD'] = df['CVD'] / df['5d_sma_CVD']

    # Awesome Oscillator (AO)
    df['5d_sma_mid'] = df['price'].rolling(window=5).mean()
    df['34d_sma_mid'] = df['price'].rolling(window=34).mean()
    df['AO'] = df['5d_sma_mid'] - df['34d_sma_mid']

    # Accelerator Oscillator (AC)
    df['5d_sma_AO'] = df['AO'].rolling(window=5).mean()
    df['AC'] = df['AO'] - df['5d_sma_AO']

    # z-score of Shill score
    df['3d_change_shill']  = df['shill_score'].diff(3)
    rolling_mean_cvd = df['3d_change_shill'].rolling(window=30).mean()
    rolling_std_cvd = df['3d_change_shill'].rolling(window=30).std()
    df['z_score_3d_change_shill'] = (df['3d_change_shill'] - rolling_mean_cvd) / rolling_std_cvd
    
    # log shill score
    df['log_shill_score'] = np.where(df['shill_score'] > 0, np.log(df['shill_score']), np.where(df['shill_score'] < 0, -np.log(np.abs(df['shill_score'])), 0))
    
    #AO shill score
    df['5d_sma_shill'] = df['shill_score'].rolling(window=5).mean()
    df['34d_sma_shill'] = df['shill_score'].rolling(window=34).mean()
    df['AO_shill'] = df['5d_sma_shill'] / df['34d_sma_shill'] 
    
    # Bollinger
    # Calculate Middle Band
    df['20d_SMA'] = df['price'].rolling(window=20).mean()
    # Calculate Standard Deviation over the same period
    df['20d_STD'] = df['price'].rolling(window=20).std()
    # Calculate Upper and Lower Bands
    df['Upper_Band'] = df['20d_SMA'] + (df['20d_STD'] * 2)
    df['Lower_Band'] = df['20d_SMA'] - (df['20d_STD'] * 2)
    # Optional: Calculate Bollinger Band Width

    #features
    df['BB_Width'] = (df['Upper_Band'] - df['Lower_Band']) / df['20d_SMA']
    df['P_vs_upper'] = df['price'] / df['Upper_Band']
    df['P_vs_lower'] = df['price'] / df['Lower_Band']
    
    #Drop unnecessary columns
    df.drop(columns=['5d_sma_AO', '5d_sma_mid', '34d_sma_mid', '5d_sma_AO', '3d_change_shill'], inplace=True)
    #'5d_sma_shill', '34d_sma_shill'
    
    df['14d_rtn'] = (df['price'].shift(-14) - df['price'])/ df['price']
    df['7d_rtn'] = (df['price'].shift(-7) - df['price'])/ df['price']
    
    #fillna
    df['shill_score'] = df['shill_score'].fillna(0)
    df['avg_sentiment_score'] = df['avg_sentiment_score'].fillna(0)
    df['num_mentions'] = df['num_mentions'].fillna(0)
    df['shill_score_7d_ema'] = df['shill_score'].ewm(span=7, adjust=False).mean()
    df['avg_sentiment_score_7d_ema'] = df['avg_sentiment_score'].ewm(span=7, adjust=False).mean()
    df['num_mentions_7d_ema'] = df['num_mentions'].ewm(span=7, adjust=False).mean()

    return df   

#The below pulls from cg for the shill chart
def coin_df_cg(days, coin, df):

    grouped_telegram_comments_data = group_data(df)

    coin_data = grouped_telegram_comments_data[grouped_telegram_comments_data['coin_mentions'].apply(lambda x: coin.upper() in x)]

    #now trying to see rolling dates, with some entries empty so can compare to eg price data
    #build new df with date starting 14 July, ending on last post
    all_dates = pd.DataFrame(pd.date_range(start='2020-07-01', end=coin_data['date'].max()), columns=['date'])

    #merge comments and price data
    merged_coin_data = pd.merge(all_dates, coin_data, how='left', on='date')

    # create shill score
    merged_coin_data['shill_score'] = merged_coin_data['num_mentions'] * merged_coin_data['avg_sentiment_score']

    # fillna
    merged_coin_data['num_mentions'] = merged_coin_data['num_mentions'].fillna(0)
    merged_coin_data['shill_score'] = merged_coin_data['shill_score'].fillna(0)
    merged_coin_data['avg_sentiment_score'] = merged_coin_data['avg_sentiment_score'].fillna(0)
     
    df_price = pull_prices_cg(days, coin)
     
    merged_coin_prices = df_price   
    
    
    start_date = datetime.strptime('2020-01-01', '%Y-%m-%d')

    # Filter the dataframes
    filtered_coin_data = merged_coin_data[merged_coin_data['date'] >= start_date]
    filtered_price_data = merged_coin_prices[merged_coin_prices['Date'] >= start_date]
    
    #rename date col
    filtered_price_data = filtered_price_data.rename({'Date':'date'}, axis=1)

    #merge
    df = pd.merge(filtered_price_data, filtered_coin_data, how='left', on='date')
    
    
    # Technicals
    df['20d_ema'] = df['price'].ewm(span=20, adjust=False).mean()
    df['10d_ema'] = df['price'].ewm(span=10, adjust=False).mean()
    df['5d_ema'] = df['price'].ewm(span=5, adjust=False).mean()
    
    
    df['10_20_ema'] = df['10d_ema'] / df['20d_ema'] 
    df['5_10_ema'] = df['5d_ema'] / df['10d_ema'] 
    
    df['10_20_acc'] = df['10_20_ema'] / df['10_20_ema'].rolling(window=5).mean()
    df['5_10_acc'] = df['5_10_ema'] / df['5_10_ema'].rolling(window=5).mean()

    # 14 day RSI
    delta = df['price'].diff(1)
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['14d_RSI'] = 100 - (100 / (1 + rs))
    
    # 7 day RSI 
    delta = df['price'].diff(1)
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/7, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/7, adjust=False).mean()
    rs = gain / loss
    df['7d_RSI'] = 100 - (100 / (1 + rs))

    # Calculate CVD
    df['price_change'] = df['price'].diff()
    df['volume_direction'] = df['24h_volume'] * df['price_change'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    df['CVD'] = df['volume_direction'].cumsum()

    # Drop columns
    df.drop(columns=['price_change', 'volume_direction'], inplace=True)
    
    # Z-Score of 3-Day Change in CVD
    df['3d_change_CVD'] = df['CVD'].diff(3)
    rolling_mean = df['3d_change_CVD'].rolling(window=30).mean()
    rolling_std = df['3d_change_CVD'].rolling(window=30).std()
    df['z_score_3d_change_CVD'] = (df['3d_change_CVD'] - rolling_mean) / rolling_std
    
    # AC of CVD - similar concept to z-score
    df['5d_sma_CVD'] = df['CVD'].rolling(window=5).mean()
    df['AC_CVD'] = df['CVD'] / df['5d_sma_CVD']

    # Awesome Oscillator (AO)
    df['5d_sma_mid'] = df['price'].rolling(window=5).mean()
    df['34d_sma_mid'] = df['price'].rolling(window=34).mean()
    df['AO'] = df['5d_sma_mid'] - df['34d_sma_mid']

    # Accelerator Oscillator (AC)
    df['5d_sma_AO'] = df['AO'].rolling(window=5).mean()
    df['AC'] = df['AO'] - df['5d_sma_AO']

    # z-score of Shill score
    df['3d_change_shill']  = df['shill_score'].diff(3)
    rolling_mean_cvd = df['3d_change_shill'].rolling(window=30).mean()
    rolling_std_cvd = df['3d_change_shill'].rolling(window=30).std()
    df['z_score_3d_change_shill'] = (df['3d_change_shill'] - rolling_mean_cvd) / rolling_std_cvd
    
    # log shill score
    df['log_shill_score'] = np.where(df['shill_score'] > 0, np.log(df['shill_score']), np.where(df['shill_score'] < 0, -np.log(np.abs(df['shill_score'])), 0))
    
    #AO shill score
    df['5d_sma_shill'] = df['shill_score'].rolling(window=5).mean()
    df['34d_sma_shill'] = df['shill_score'].rolling(window=34).mean()
    df['AO_shill'] = df['5d_sma_shill'] / df['34d_sma_shill'] 
    
    # Bollinger
    # Calculate Middle Band
    df['20d_SMA'] = df['price'].rolling(window=20).mean()
    # Calculate Standard Deviation over the same period
    df['20d_STD'] = df['price'].rolling(window=20).std()
    # Calculate Upper and Lower Bands
    df['Upper_Band'] = df['20d_SMA'] + (df['20d_STD'] * 2)
    df['Lower_Band'] = df['20d_SMA'] - (df['20d_STD'] * 2)
    # Optional: Calculate Bollinger Band Width

    #features
    df['BB_Width'] = (df['Upper_Band'] - df['Lower_Band']) / df['20d_SMA']
    df['P_vs_upper'] = df['price'] / df['Upper_Band']
    df['P_vs_lower'] = df['price'] / df['Lower_Band']

    
    #Drop unnecessary columns
    df.drop(columns=['5d_sma_AO', '5d_sma_mid', '34d_sma_mid', '5d_sma_AO', '3d_change_shill'], inplace=True)
    #'5d_sma_shill', '34d_sma_shill'
    
    df['14d_rtn'] = (df['price'].shift(-14) - df['price'])/ df['price']
    df['7d_rtn'] = (df['price'].shift(-7) - df['price'])/ df['price']
    
    #fillnas
    df['shill_score'] = df['shill_score'].fillna(0)
    df['avg_sentiment_score'] = df['avg_sentiment_score'].fillna(0)
    df['num_mentions'] = df['num_mentions'].fillna(0)
    df['shill_score_7d_ema'] = df['shill_score'].ewm(span=7, adjust=False).mean()
    df['avg_sentiment_score_7d_ema'] = df['avg_sentiment_score'].ewm(span=7, adjust=False).mean()
    df['num_mentions_7d_ema'] = df['num_mentions'].ewm(span=7, adjust=False).mean()

    return df  

def shill_chart(df):
    # Plot the data
    plt.style.use('dark_background')
    fig, ax1 = plt.subplots(figsize=(10, 6))

    #color = 'tab:blue'

    # Pastel colors for the plots
    pastel_blue = '#72A1E5'
    pastel_green = '#B8DFD8'
    neon_pink = '#FF6EC7'

    title_font = {'family': 'sans-serif',
                  'color': 'white',
                  'weight': 'bold',
                  'size': 14,
                 }


    ax1.set_xlabel('Date')
    ax1.set_ylabel('Number of Mentions', color=pastel_blue)
    ax1.scatter(df['date'], df['num_mentions'], color=pastel_blue, s=10)
    ax1.tick_params(axis='y', labelcolor=pastel_blue)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Shill Score', color=neon_pink)
    ax2.scatter(df['date'], df['shill_score'], color=neon_pink, s=10)
    ax2.tick_params(axis='y', labelcolor=neon_pink)

    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60)) # Move the right y-axis to the right a bit
    ax3.set_ylabel('Price in USD', color=pastel_green)

    # Use plot function for price data instead of scatter
    ax3.plot(df['date'], df['price'], color=pastel_green)
    ax3.tick_params(axis='y', labelcolor=pastel_green)

    fig.autofmt_xdate()
    fig.tight_layout()
    plt.title("Coin mentions and shill score over time", fontdict=title_font)

    ax1.grid(False)
    ax2.grid(False)
    ax3.grid(False)

    plt.show()
    st.pyplot(fig)

def current_cluster(df):
    features_for_prediction = ['num_mentions', 'avg_sentiment_score', 'shill_score', '10_20_ema', '10_20_acc', '14d_RSI', '3d_change_CVD', 'AO', 'AC_CVD',
           'shill_score_7d_ema', 'avg_sentiment_score_7d_ema', 'num_mentions_7d_ema', 'BB_Width', 'P_vs_lower']
    most_recent_data = df.tail(1)[features]
    most_recent_data = most_recent_data.fillna(0)
    most_recent_data_sc = sc.fit_transform(most_recent_data)
    cluster_label = km.predict(most_recent_data_sc)
    print("The cluster label for the most recent data is:", cluster_label)
    return cluster_label


def boxplots(df):

    average_14d_rtn_per_cluster = df.groupby('labels')['14d_rtn'].mean().sort_values(ascending=False)

    features_for_plotting = ['14d_rtn', '14d_RSI', '10_20_ema', '10_20_acc', 'AC_CVD',
            'shill_score_7d_ema', 'avg_sentiment_score_7d_ema', 'num_mentions_7d_ema', 'BB_Width']

    fig = plt.figure(figsize=(20, 20))

    # Determine the order of clusters based on 14d_rtn
    cluster_order = average_14d_rtn_per_cluster.index.tolist()

    # Create a 4x4 grid of subplots
    for i, feature in enumerate(features_for_plotting, 1):
        plt.subplot(3, 3, i)
        sns.boxplot(x='labels', y=feature, data=ZZ_1, order=cluster_order)
        plt.title(feature)

    plt.tight_layout()
    plt.show()
    st.pyplot(fig)


#This should be redundant
#def cluster_return(df):
#    features_for_prediction = ['num_mentions', 'avg_sentiment_score', 'shill_score', '10_20_ema', '10_20_acc', '14d_RSI', '7d_RSI', '3d_change_CVD', 'z_score_3d_change_CVD', 'AO', 'AC', 'AC_CVD', 'z_score_3d_change_shill', 'log_shill_score',
#                            'shill_score_7d_ema', 'avg_sentiment_score_7d_ema', 'num_mentions_7d_ema']
#    most_recent_data = new_df.tail(1)[features]
#    most_recent_data = most_recent_data.fillna(0)
#    most_recent_data_sc = sc.fit_transform(most_recent_data)
#    cluster_label = km.predict(most_recent_data_sc)
#    return cluster_label


def create_labels(df):
    features = ['num_mentions', 'avg_sentiment_score', 'shill_score', '10_20_ema', '10_20_acc', '14d_RSI', '3d_change_CVD', 'AO', 'AC_CVD',
            'shill_score_7d_ema', 'avg_sentiment_score_7d_ema', 'num_mentions_7d_ema', 'BB_Width', 'P_vs_lower']
    ZZ = df[:-14][features]
    ZZ=ZZ.fillna(0)
    #ZZ.isnull().sum()
    km = KMeans(n_clusters=5)
    sc = StandardScaler()
    ZZ_sc = sc.fit_transform(ZZ)
    km.fit(ZZ_sc)
    from sklearn.metrics import silhouette_score
    #silhouette_score(ZZ_sc, km.labels_)
    ZZ_1 = ZZ.copy()
    ZZ_1['labels'] = km.labels_
    #ZZ_1['labels'].value_counts()
    ZZ_1['14d_rtn'] = df['14d_rtn'][:-14]
    #Adding new lines
    ZZ_1['date'] = df['date'][:-14]
    ZZ_1['price'] = df['price'][:-14]
    ZZ_group = ZZ_1.groupby('labels').mean().sort_values(by='14d_rtn', ascending=False)
    return ZZ_1

def create_labels_recent(df):
    features = ['num_mentions', 'avg_sentiment_score', 'shill_score', '10_20_ema', '10_20_acc', '14d_RSI', '3d_change_CVD', 'AO', 'AC_CVD',
            'shill_score_7d_ema', 'avg_sentiment_score_7d_ema', 'num_mentions_7d_ema', 'BB_Width', 'P_vs_lower']
    ZZ = df[features]
    ZZ=ZZ.fillna(0)
    #ZZ.isnull().sum()
    km = KMeans(n_clusters=5, random_state=42)
    sc = StandardScaler()
    ZZ_sc = sc.fit_transform(ZZ)
    km.fit(ZZ_sc)
    from sklearn.metrics import silhouette_score
    #silhouette_score(ZZ_sc, km.labels_)
    ZZ_1 = ZZ.copy()
    ZZ_1['labels'] = km.labels_

    #Adding new lines
    ZZ_1['date'] = df['date']
    ZZ_1['price'] = df['price']

    return ZZ_1 

def cluster_return(df):
    features = ['num_mentions', 'avg_sentiment_score', 'shill_score', '10_20_ema', '10_20_acc', '14d_RSI', '7d_RSI', '3d_change_CVD', 'z_score_3d_change_CVD', 'AO', 'AC', 'AC_CVD', 'z_score_3d_change_shill', 'log_shill_score',
                            'shill_score_7d_ema', 'avg_sentiment_score_7d_ema', 'num_mentions_7d_ema']
    
    sc = StandardScaler()
    km = KMeans(n_clusters=5, random_state=42)
    most_recent_data_sc = sc.fit_transform(most_recent_data)
    km.fit(most_recent_data_sc)
    most_recent_data = df.tail(1)[features]
    most_recent_data = most_recent_data.fillna(0)
    cluster_label = km.predict(most_recent_data_sc)
    return cluster_label



def find_ref(lst, top_cluster):
    # Search for the label in reverse
    target_label = top_cluster
    days_ago = None
    for i, label in enumerate(reversed(lst)):
        if label == target_label:
            days_ago = i
            break

    if days_ago is not None:
        return i
    else:
        return None




def build_landing(n, coins_list, df):
    pd.set_option('display.float_format', '{:.3f}'.format)
    df_50 = pd.DataFrame()

    for i, ticker in enumerate(coins_list_ex_usdt['Ticker'].head(n)):
        ticker = ticker.upper()
        coin_data = coin_df(1100, ticker, df)
        last_row = coin_data.tail(1)
        
        cluster = create_labels_recent(coin_data)
        cluster = cluster['labels'].iloc[-1]
        
        #find top cluster
        x = create_labels(coin_data)
        top_cluster = x.groupby('labels').mean().sort_values(by='14d_rtn', ascending=False).index[0]
        
        last_7 = create_labels_recent(coin_data)
        last_7 = list(last_7['labels'][-7:])
        wen_top_cluster = find_ref(last_7, top_cluster)
        
        df_50 = pd.concat([df_50, last_row], ignore_index=True)
        df_50.at[i, 'coin'] = ticker
        df_50.at[i, 'cluster'] = cluster
        df_50.at[i, 'top_cluster'] = top_cluster
        df_50.at[i, 'wen_top'] = wen_top_cluster
        time.sleep(5)
        
    features = ['coin', 'cluster', 'top_cluster', 'wen_top', 'num_mentions','avg_sentiment_score', 'shill_score', 'num_mentions_7d_ema', 'avg_sentiment_score_7d_ema','shill_score_7d_ema', 'AO_shill', '14d_RSI', '10_20_ema', '10_20_acc', 'AC_CVD', 'P_vs_upper', 'P_vs_lower']
    df_50 = df_50[features]
    df_50 = df_50.rename(columns={'cluster': 'clust', 'num_mentions': '#mentions', 'avg_sentiment_score':'sent', 'shill_score': 'shill', 'num_mentions_7d_ema': '#mentions_7d', 'avg_sentiment_score_7d_ema': 'sent7d' ,'shill_score_7d_ema': 'shill7d', '14d_RSI':'RSI', 'P_vs_upper':'P/upper', 'P_vs_lower':'P/lower'})
    df_50.set_index('coin', inplace=True)

    return df_50
    st.dataframe(df_50)
#    st.pyplot(fig)
    

def create_labels(df):
    features = ['num_mentions', 'avg_sentiment_score', 'shill_score', '10_20_ema', '10_20_acc', '14d_RSI', '3d_change_CVD', 'AO', 'AC_CVD',
            'shill_score_7d_ema', 'avg_sentiment_score_7d_ema', 'num_mentions_7d_ema', 'BB_Width', 'P_vs_lower']
    ZZ = df[:-14][features]
    ZZ=ZZ.fillna(0)
    #ZZ.isnull().sum()
    km = KMeans(n_clusters=5, random_state=42)
    sc = StandardScaler()
    ZZ_sc = sc.fit_transform(ZZ)
    km.fit(ZZ_sc)
    from sklearn.metrics import silhouette_score
    #silhouette_score(ZZ_sc, km.labels_)
    ZZ_1 = ZZ.copy()
    ZZ_1['labels'] = km.labels_
    #ZZ_1['labels'].value_counts()
    ZZ_1['14d_rtn'] = df['14d_rtn'][:-14]
    #Adding new lines
    ZZ_1['date'] = df['date'][:-14]
    ZZ_1['price'] = df['price'][:-14]
    return ZZ_1






#Functions end here

#streamlit function




###################################################


#Pull crypto list
try:
    cryptocurrencies_list = 'complete_coins_list.csv'
except:
    data = fetch_cryptocurrencies()
    if data:
        save_to_csv_using_dataframe(data, 'complete_coins_list.csv')
    
cryptocurrencies_list = 'complete_coins_list.csv'
coins_list = pd.read_csv(cryptocurrencies_list)
coins_list = coins_list[coins_list['24h Volume'] > 500000]
crypto_symbols_set = set(coins_list['Ticker'].str.upper()) # faster lookup


#def color_df(val):
#    if val > 7.5:
#        color = 'green'
#    elif val < 7.5:
#        color = 'red'
#    return f'bacground-color: {color}'
#Printing the main list

#Choose coin, default SOL
coin_name = 'SOL'
coin_name = st.sidebar.selectbox(
    'Select coin',
    ('TON', 'SOL', 'BNB', 'ATOM', 'AAVE', 'SNX', 'ADA', 'LINK', 'MKR', 'GMX',
    'RNDR', 'AKT', 'AVAX', 'TIA', 'STRD', 'LDO', 'FXS', 'OP', 'ARB', 'UNI', 'RON', 'IMX', 
    'MATIC', 'ETH', 'BTC', 'WIF', 'DOGE', 'BONK', 'DOT', 'XRP', 'EOS', 'DYDX', 'GMX')
)


#coins_list_ex_usdt = coins_list[coins_list['Ticker'] != 'usdt']

# List of tickers to exclude - kept the old name ex_usdt for ease
excluded_tickers = ['usdt', 'steth', 'usdc']
coins_list_ex_usdt = coins_list[~coins_list['Ticker'].isin(excluded_tickers)]



#Load master doc with master sentiment
df = pd.read_csv('Master_sentiment_new.csv') #NB this is the latest file with all comments run through HF

df['date'] = pd.to_datetime(df[' message.date']).dt.date

#First I want to see a chart with all coins mentioned in alst week and number of mentions, shill score on the axis
 

#NB new function which explodes the coin refs
#grouped_telegram_comments_data = shillometer_charts(df)
#st.dataframe(grouped_telegram_comments_data.tail(20))
#st.dataframe(grouped_telegram_comments_data_new.style.applymap(color_df, subset=['shill_score']))

#New shillometer charts

def shillometer_charts(df):
    # Convert 'coin_mentions' from string representation of list to actual list
    df['coin_mentions'] = df['coin_mentions'].apply(eval)

    # Explode 'coin_mentions' so each coin mention gets its own line
    exploded_telegram_comments_df = df.explode('coin_mentions')

    # Group by date and then coin mentions and add num_mentions as count of message.text and 
    # avg_sentiment as mean sentiment score. Then reset index
    grouped_telegram_comments_data = exploded_telegram_comments_df.groupby(['date', 'coin_mentions']).agg(
        num_mentions=pd.NamedAgg(column="message.text", aggfunc="count"),
        avg_sentiment_score=pd.NamedAgg(column="sentiment_score", aggfunc="mean")
    ).reset_index()

    # Filter out rows with no coin mentions
    grouped_telegram_comments_data = grouped_telegram_comments_data[grouped_telegram_comments_data['coin_mentions'].notna()]

    grouped_telegram_comments_data['date'] = pd.to_datetime(grouped_telegram_comments_data['date'])

    # Calculate 'Shill score'
    grouped_telegram_comments_data['Shill_score'] = grouped_telegram_comments_data['num_mentions'] * grouped_telegram_comments_data['avg_sentiment_score']

    return grouped_telegram_comments_data

# Load data
#df = pd.read_csv('Master_sentiment.csv')
#df['date'] = pd.to_datetime(df[' message.date']).dt.date

# Apply the grouping function
grouped_telegram_comments_data = shillometer_charts(df)
most_recent_day = grouped_telegram_comments_data['date'].max()
recent_data = grouped_telegram_comments_data[grouped_telegram_comments_data['date'] == most_recent_day]
recent_data_sorted_desc = recent_data.sort_values(by='Shill_score', ascending=False)

# Plot column chart for the most recent day
#fig = plt.figure(figsize=(10,6))
#plt.bar(recent_data_sorted_desc['coin_mentions'], recent_data_sorted_desc['Shill_score'])
#plt.xlabel('Coins')
#plt.ylabel('Shill Score')
#plt.title('Shill Score by Coin for the Most Recent Day')
#plt.xticks(rotation=90) # Rotate the x-axis labels to show them clearly
#plt.show()
#st.pyplot(fig)

# Apply a style to use for the chart
plt.style.use('dark_background')

# Plot column chart for the most recent day
fig, ax = plt.subplots(figsize=(10,6))
bars = ax.bar(recent_data_sorted_desc['coin_mentions'], recent_data_sorted_desc['Shill_score'])
ax.set_xlabel('Coins', color='white')
ax.set_ylabel('Shill Score', color='white')
ax.set_title("Today's Shill Score by Coin", color='white')
ax.tick_params(axis='x', rotation=90, colors='white') # Rotate the x-axis labels and set their color to white
ax.tick_params(axis='y', colors='white') # Set y-axis tick colors to white

# You can set the edge color of the bars if needed
for bar in bars:
    bar.set_edgecolor('white')

# Show the plot in Streamlit
st.pyplot(fig)




# Aggregate data by week
grouped_telegram_comments_data['week'] = grouped_telegram_comments_data['date'].dt.isocalendar().week
weekly_data = grouped_telegram_comments_data.groupby(['week', 'coin_mentions']).agg(
    total_mentions=pd.NamedAgg(column="num_mentions", aggfunc="sum"),
    avg_weekly_sentiment_score=pd.NamedAgg(column="avg_sentiment_score", aggfunc="mean")
).reset_index()

# Calculate 'Weekly Shill score'
weekly_data['Weekly Shill score'] = weekly_data['total_mentions'] * weekly_data['avg_weekly_sentiment_score']

# Get the data for the most recent week
most_recent_week = weekly_data['week'].max()
recent_weekly_data = weekly_data[weekly_data['week'] == most_recent_week]
recent_weekly_data_sorted_desc = recent_weekly_data.sort_values(by='Weekly Shill score', ascending=False)

# Plot bar chart for the most recent wee
fig = plt.figure(figsize=(10,6))
plt.bar(recent_weekly_data_sorted_desc['coin_mentions'], recent_weekly_data_sorted_desc['Weekly Shill score'])
plt.xlabel('Coins')
plt.ylabel('Weekly Shill Score ')
plt.title('Weekly Shill Score by Coin')
plt.xticks(rotation=90, fontsize=7) # Rotate the x-axis labels to show them clearly
plt.show()
st.pyplot(fig)

#st.markdown("""
### Below, we show the top 50 coins by market cap. The table is designed to highlight coins which have triggered a top cluster today

### This is 'wen_top' = 0. If the coin has triggered a top cluster in the last week, wen_top shows you how many days ago

#The other metrics are:

#- 7d ema of average sentiment, shill score, number of mentions
#- AO shill = 5 dma shill score / 34 dma shill score - this gives a sense of magnitude and is designed to act like a z-score
#- 14d RSI - main momentum indicator 
#- 10_20_ema - 10day ema / 20 day ema - again a short term momentum indicator
#- 10_20_acc - 10_20_ema divided by its own 5 day ema - this is designed to pickup early changes in momentum
#- AC_CVD - CVD (cumulative volume delta, simlar to OBV ie a rough metric of price*volume) / 5 day average of CVD - this is basically a proxy on the pickup of short term bullish or bearish volume
#- P_vs_upper - price / upper Bollinger Band
#- P_vs_lower - price / lower Bollinger band
#
#""")


#landing = build_landing(25, coins_list_ex_usdt, df)
#st.dataframe(landing)



####################

#Now the coin specific section

st.write(f'## Here are the charts for {coin_name}')



#Load new df with specific coin
new_df = coin_df_cg(1150, coin_name, df)

#st.write(new_df.head())

#Load shill chart
shill_chart(new_df)

#K means section

st.markdown("""
### For each coin, we cluster historically (eg on certain days it will be in cluster x, let's say the high shill score cluster or the high RSI cluster), and below is the mean feature value by cluster, ranked by 14 day forward return

#
""")

#ZZ_1 = create_labels(new_df)
#ZZ_group = ZZ_1.groupby('labels').mean().sort_values(by='14d_rtn', ascending=False)
#st.dataframe(ZZ_group)


#Load boxplots
#boxplots(ZZ_1)

#Work out current cluster
#cluster = cluster_return(new_df)

#cluster = create_labels_recent(coin_name)
#cluster = cluster['labels'].iloc[-1]
#st.write(f'The most recent cluster for {coin_name} is {cluster}')

#Want to show in the plot if top signal has fired that week

#print graph for most recent cluster
#label_return_final(ZZ_1)

st.markdown("""
### We then run a strategy of going long at each cluster signal and holding for 28 days, with a 10 percent stoploss
You can see that there are certain clusters than consistently outperform and in most cases this is because the model sniffs out early momentum signals
In the table, you can see the distribution of cluster returns, with number of triggers and hit rate.

""")
#OLD
#table = label_return_final(ZZ_1)
#st.dataframe(table)

df_pca_kmeans = kmeans_pca(new_df)
table2 = label_return_final(df_pca_kmeans)
st.dataframe(table2)

#df_pca_group = df_pca_kmeans.groupby('labels').mean().sort_values(by='14d_rtn', ascending=False)
#st.dataframe(df_pca_group)

st.markdown("""
### You can see the buy triggers over time in the chart below 
""")

# Need to add back the fwd returns but do a fillna
#buy_signals_over_time(new_df, ZZ_1)

buy_signals_over_time(new_df, df_pca_kmeans)


#current_cluster(df_pca_kmeans)





#st.markdown("""
### Lastly, we backtest this strategy with a time based train test split

#This works by splitting the data into 30 test periods (roughly 2-4 weeks given the training period) and at each point, running the PCA and k-means algorithm and picking the most successful historical cluster, then holding for 28 days with a 10 percent stop loss

# 
#""")
#full_test_data = dynamic_label_return_strategy(df_pca_kmeans)

#dynamic_label_return_strategy_plot(full_test_data)






