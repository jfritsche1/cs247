import re
from collections import defaultdict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def Plot_Word_Frequency(data, field, type):
    #The code in this cell is modified from https://www.kaggle.com/wordcards/stock-market-tweets-wordcloud
    ticker_pattern = re.compile(r'(^\$[A-Z]+|^\$ES_F)', flags=re.IGNORECASE)
    ht_pattern = re.compile(r'#\w+', flags=re.IGNORECASE)
    ticker_dic = defaultdict(int)
    ht_dic = defaultdict(int)

    for i, row in data.iterrows():
        text = data.loc[i,field]

        for word in text.split():
            word = word.upper()
            if ticker_pattern.fullmatch(word) is not None:
                ticker_dic[word[1:]] += 1
                
            word = word.lower()
            if ht_pattern.fullmatch(word) is not None:
                ht_dic[word] += 1
    ticker_df = pd.DataFrame.from_dict(
        ticker_dic, orient='index').rename(columns={0:'count'})\
        .sort_values('count', ascending=False).head(20)
        
    ht_df = pd.DataFrame.from_dict(
        ht_dic, orient='index').rename(columns={0:'count'})\
        .sort_values('count', ascending=False).head(20)

    fig, ax = plt.subplots(1, 2, figsize=(12,8))
    plt.suptitle('Frequent Tickers and Hashtags for '+ type, fontsize=16)
    plt.subplots_adjust(wspace=0.4)

    sns.barplot(x=ticker_df['count'], y=ticker_df.index, orient='h', ax=ax[0])
    ax[0].set_title('Top 20 Tickers')

    sns.barplot(x=ht_df['count'], y=ht_df.index, orient='h', ax=ax[1])
    ax[1].set_title('Top 20 HashTags')

    plt.show()