import seaborn as sns
import matplotlib.pyplot as plt

def PrintSummaryStatistics(stock_data, tickers):
    #We will use Adjusted Closing Price for the analysis
    #Adjusted closing price is the closing price after adjustments for all applicable splits and dividend distributions.

    for key,value in tickers.items():
        data = stock_data[stock_data['Ticker'] == key]
        print("Adjusted Closing Price for ", value)
        print(data['Adj Close'].describe())
        print("*"*50)

def AdjCloseBoxPlot(stock_data):
    #Since price for Amazon stock and Google stock are much higher, the plots are shown on Log Scale so we can visualize all in one plot
    #Boxplot for Log Scaled Adjusted Closing Price"
    sns.set(rc={'figure.figsize':(20,20)})
    ax = sns.boxplot(x="Ticker", y="Adj Close", data=stock_data, palette='Pastel2')
    ax.set_yscale("log")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.title("Boxplot for Log Scaled Adjusted Closing Price")
    
    None
#%%
def AdjClosePricePerCategoryBoxPlot(stock_data, categories):
    #Now we look at the boxplots of adjusted closing price for companites in each category separately
    #Log scale is not applied here, so we can see the actual price.
    for key,value in categories.items():
        data = stock_data[stock_data["Ticker"].isin(value)]
        sns.set(rc={'figure.figsize':(10,10)})
        ax = sns.boxplot(x="Ticker", y="Adj Close", data=data, palette='Pastel2')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        plt.title("Boxplot for Adjusted Closing Prices for the Companies in '" + key + "' Category")
        plt.show()
        print('\n')
        None
#%%
def AdjClosePricePerCategoryDaily(stock_data, categories, tickers):
    #Now, we look at the stock charts for the companites in each category
    #There are some similarlities in movements among the same category

    for key,value in categories.items():
        print("*"*130)
        print(""," "*52, "Category - " , key)
        print("*"*130)
        for ticker in value:
            data = stock_data[stock_data['Ticker'] == ticker]
            sns.set(rc={'figure.figsize':(17,5)})
            ax = sns.lineplot(data=data, x="Date", y="Adj Close", color = 'cornflowerblue')
            ax = sns.lineplot(data=data, x="Date", y="High", color='mediumpurple')
            ax = sns.lineplot(data=data, x="Date", y="Low", color = "lightgreen")
            ax.set(xlabel='Date', ylabel='Stock Price')
            plt.setp(ax.get_xticklabels(), rotation=80)
            plt.title(tickers[ticker])
            plt.legend(["Adj Closing Price", "Highest Price", "Lowest Price"], loc ="lower right")
            plt.show()
            print("\n")