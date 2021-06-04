import numpy as np
import pandas as pd

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#%%
#The Compound score is a metric that is scaled between -1 and +1
#-1 being most negative score and +1 being most positive score

def vader_sentiment(df, columnName):
    #Sentiment Analysis using Vader
    analyzer = SentimentIntensityAnalyzer()
    scores = []
    # Declare variables for scores
    text = np.array(df[columnName])
    for i in range(text.shape[0]):
        compound = analyzer.polarity_scores(text[i])["compound"]
        pos = analyzer.polarity_scores(text[i])["pos"]
        neu = analyzer.polarity_scores(text[i])["neu"]
        neg = analyzer.polarity_scores(text[i])["neg"]
        
        scores.append({"Compound": compound,
                            "Positive": pos,
                            "Negative": neg,
                            "Neutral": neu
                        })
    sentiments_score = pd.DataFrame.from_dict(scores)
    twitter_sentiments_score = pd.concat([df.reset_index(drop=True),sentiments_score.reset_index(drop=True)], axis=1)

    return twitter_sentiments_score