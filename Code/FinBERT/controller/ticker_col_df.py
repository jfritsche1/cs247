##############################################################################################
###This code is necessary ONLY IF we are serarating the tweets into each indivisual tickers###
##############################################################################################

#Create a new dataframe with columns for each stock symbol.

#If the tweet contains the following ticker symbols, the column for the ticker symbol is 1, otherwise 0.
#$MSFT, $AAPL, $AMZN, $FB, $BRK.B, $GOOG, $JNJ, $JPM, $V, $PG, $MA, $INTC $UNH, $BAC, $T, $HD, $XOM, $DIS, $VZ, $KO, $MRK, $CMCSA, $CVX, $PEP, $PFE

#It coniders ticker symbol only, and not the words that are related.  
#For example, a tweet has a word "Amazon" but not "$AMZN", the column "AMZN" will stay 0.

def ticker_col_dataframe(data, tickers, field):
  new_df = data.copy()
  for symbol in tickers.keys():
    new_df[symbol] = 0

  for i, row in data.iterrows():
    text = data.loc[i,field]
    if "$BRK.B" in text:
      sp_char =  [",","?",":",";","'","-","!"]
    else:
      sp_char =  [",",".","?",":",";","'","-","!"]
    for char in sp_char:
      text = text.replace(char,' ')
      
    for word in text.split():
        word = word.upper()
        for symbol in tickers.keys():
          t = '$' + symbol
          if t == word:
            new_df.loc[i,symbol] = 1
  return new_df