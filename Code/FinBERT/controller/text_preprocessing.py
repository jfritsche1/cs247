import time
import re
import pandas as pd
import preprocessor as p

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

#The code in this cell is modified from https://towardsdatascience.com/basic-tweet-preprocessing-in-python-efd8360d529e

def text_preprocessing(d, columnName, field):
    tic = time.perf_counter()
    data = d.copy()
    data[columnName] = data[field]
    for i, row in data.iterrows():
        #$DIS is considered as plural of DI after removing '$'
        #So, replace $DIS with $DISNEY
        data.loc[i,columnName] = data.loc[i,columnName].replace("$DIS","$DISNEY")
        data.loc[i,columnName] = p.clean(data.loc[i,columnName])
  
    def preprocess_data(data):
        #Removes Numbers
        data = data.astype(str).str.replace('\d+', '')
        lower_text = data.str.lower()
        lemmatizer = WordNetLemmatizer()
        w_tokenizer =  TweetTokenizer()

        def lemmatize_text(text):
            return [(lemmatizer.lemmatize(w)) for w in w_tokenizer.tokenize((text))]
        def remove_punctuation(words):
            new_words = []
            for word in words:
                new_word = re.sub(r'[^\w\s]', '', (word))
                if new_word != '':
                    new_words.append(new_word)
            return new_words
        words = lower_text.apply(lemmatize_text)
        words = words.apply(remove_punctuation)
        return pd.DataFrame(words)

    pre_tweets = preprocess_data(data[columnName])
    data[columnName] = pre_tweets
    stop_words = set(stopwords.words('english'))
    data[columnName] = data[columnName].apply(lambda x: [item for item in x if item not in stop_words])

    for i, row in data.iterrows():
        data.loc[i,columnName] = ' '.join(data.loc[i,columnName])
    toc = time.perf_counter()
    print(f" *** Text Pre-Processing Complete: {toc - tic:0.4f} seconds *** ")

    return data